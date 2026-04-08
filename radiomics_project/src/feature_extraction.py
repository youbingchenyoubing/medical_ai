import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
from tqdm import tqdm
from typing import Dict, List, Optional
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


class PyRadiomicsExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.fe_config = config['feature_extraction']
        self.sequences = config['mri_sequences']
        self.extractor = self._create_extractor()
        logger.info("PyRadiomicsExtractor initialized")
        logger.info(f"PyRadiomics settings: {self.extractor.settings}")

    def _create_extractor(self) -> featureextractor.RadiomicsFeatureExtractor:
        extractor = featureextractor.RadiomicsFeatureExtractor()

        extractor.settings['binWidth'] = self.fe_config['bin_width']
        extractor.settings['resampledPixelSpacing'] = self.fe_config['resampled_spacing']
        extractor.settings['interpolator'] = self.fe_config['interpolator']
        extractor.settings['normalize'] = self.fe_config['normalize']
        extractor.settings['force2D'] = self.fe_config['force2D']

        extractor.disableAllImageTypes()
        extractor.disableAllFeatures()

        for img_type in self.fe_config['image_types']:
            if img_type == 'Original':
                extractor.enableImageTypeByName('Original')
            elif img_type == 'LoG':
                sigma_values = self.fe_config['log_sigma_values']
                extractor.enableImageTypeByName(
                    'LoG',
                    customArgs={'sigma': sigma_values}
                )
            elif img_type == 'Wavelet':
                extractor.enableImageTypeByName('Wavelet')

        for feat_class in self.fe_config['feature_classes']:
            extractor.enableFeatureClassByName(feat_class)

        return extractor

    def extract_single_sequence(
        self,
        image_path: str,
        mask_path: str,
        sequence_name: str = ""
    ) -> Optional[Dict]:
        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)

            if image.GetSize() != mask.GetSize():
                logger.warning(
                    f"Size mismatch: image={image.GetSize()}, "
                    f"mask={mask.GetSize()}. Resampling mask to image."
                )
                mask = sitk.Resample(
                    mask, image,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor, 0,
                    mask.GetPixelID()
                )

            features = self.extractor.execute(image, mask)

            feature_dict = {}
            prefix = f"{sequence_name}_" if sequence_name else ""
            for key, value in features.items():
                if key.startswith('original_'):
                    feat_name = f"{prefix}{key}"
                    val = float(value)
                    feature_dict[feat_name] = val if not np.isnan(val) else 0.0

            logger.info(
                f"Extracted {len(feature_dict)} features from "
                f"{os.path.basename(image_path)} ({sequence_name})"
            )
            return feature_dict

        except Exception as e:
            logger.error(f"Error extracting from {image_path}: {str(e)}")
            return None

    def extract_patient_features(
        self,
        patient_dir: str,
        timepoint: str = "pre"
    ) -> Optional[Dict]:
        all_features = {}
        mask_path = os.path.join(patient_dir, f"{timepoint}_mask.nii.gz")

        if not os.path.exists(mask_path):
            logger.warning(f"Mask not found: {mask_path}")
            return None

        for seq_info in self.sequences:
            seq_code = seq_info['code']
            image_path = os.path.join(
                patient_dir, f"{timepoint}_{seq_code}.nii.gz"
            )

            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue

            features = self.extract_single_sequence(
                image_path=image_path,
                mask_path=mask_path,
                sequence_name=seq_code
            )

            if features:
                all_features.update(features)

        return all_features if all_features else None

    def extract_batch(
        self,
        processed_dir: str,
        mask_dir: str,
        output_csv: str,
        timepoint: str = "pre"
    ) -> pd.DataFrame:
        logger.info(f"Batch extracting features ({timepoint}) from {processed_dir}")

        all_features = []

        patient_dirs = sorted([
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ])

        logger.info(f"Found {len(patient_dirs)} patients")

        for patient_id in tqdm(patient_dirs, desc=f"Extracting ({timepoint})"):
            patient_dir = os.path.join(processed_dir, patient_id)

            features = self.extract_patient_features(
                patient_dir=patient_dir,
                timepoint=timepoint
            )

            if features:
                features['patient_id'] = patient_id
                features['timepoint'] = timepoint
                all_features.append(features)

        df = pd.DataFrame(all_features)

        ensure_dir(os.path.dirname(output_csv))
        df.to_csv(output_csv, index=False)

        feature_cols = [c for c in df.columns
                        if c not in ['patient_id', 'timepoint']]
        logger.info(
            f"Features saved to {output_csv}: "
            f"{len(df)} patients, {len(feature_cols)} features"
        )
        return df

    def extract_both_timepoints(
        self,
        processed_dir: str,
        output_dir: str
    ) -> Dict[str, pd.DataFrame]:
        logger.info("Extracting features for both timepoints")

        results = {}
        for timepoint in ['pre', 'post']:
            output_csv = os.path.join(
                output_dir, f"radiomics_features_{timepoint}.csv"
            )
            df = self.extract_batch(
                processed_dir=processed_dir,
                mask_dir="",
                output_csv=output_csv,
                timepoint=timepoint
            )
            results[timepoint] = df

        return results

    def get_feature_names(self) -> List[str]:
        feature_names = []

        for seq_info in self.sequences:
            seq_code = seq_info['code']
            prefix = f"{seq_code}_original_"

            for feature_class in self.fe_config['feature_classes']:
                feature_names.append(f"{prefix}{feature_class}_*")

        return feature_names

    def count_expected_features(self) -> Dict[str, int]:
        counts = {
            'firstorder': 18,
            'shape': 14,
            'glcm': 24,
            'glrlm': 16,
            'glszm': 16,
            'gldm': 14,
            'ngtdm': 5,
        }

        total_per_image_type = sum(counts.values())
        n_image_types = 1 + len(self.fe_config['log_sigma_values']) + 8
        total_per_sequence = total_per_image_type * n_image_types
        n_sequences = len(self.sequences)

        return {
            'per_image_type': total_per_image_type,
            'n_image_types': n_image_types,
            'per_sequence': total_per_sequence,
            'n_sequences': n_sequences,
            'total': total_per_sequence * n_sequences,
            'class_breakdown': counts
        }
