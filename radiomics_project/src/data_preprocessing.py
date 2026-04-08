import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


class MRIPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.target_spacing = config['preprocessing']['target_spacing']
        self.n4_config = config['preprocessing']['n4_bias_correction']
        self.reg_config = config['preprocessing']['registration']
        self.norm_config = config['preprocessing']['normalization']
        self.sequences = config['mri_sequences']
        logger.info("MRIPreprocessor initialized")

    def register_images(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        output_path: str
    ) -> sitk.Image:
        fixed_image = sitk.ReadImage(fixed_image_path)
        moving_image = sitk.ReadImage(moving_image_path)

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image,
            sitk.AffineTransform(fixed_image.GetDimension()),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=self.reg_config['histogram_bins']
        )
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(self.reg_config['sampling_percentage'])
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsGradientDescent(
            learningRate=self.reg_config['learning_rate'],
            numberOfIterations=self.reg_config['max_iterations'],
            convergenceMinimumValue=self.reg_config['convergence_min_value'],
            convergenceWindowSize=self.reg_config['convergence_window_size']
        )
        registration.SetInitialTransform(initial_transform, inPlace=False)

        final_transform = registration.Execute(fixed_image, moving_image)
        registered_image = sitk.Resample(
            moving_image, fixed_image,
            final_transform, sitk.sitkLinear, 0.0,
            moving_image.GetPixelID()
        )

        if output_path:
            ensure_dir(os.path.dirname(output_path))
            sitk.WriteImage(registered_image, output_path)
            logger.info(f"Registered image saved to: {output_path}")

        return registered_image

    def n4_bias_correction(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None
    ) -> sitk.Image:
        if not self.n4_config['enabled']:
            return image

        if mask is None:
            mask = sitk.OtsuThreshold(image, 0, 1)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(
            self.n4_config['max_iterations']
        )
        corrector.SetConvergenceThreshold(
            self.n4_config['convergence_threshold']
        )
        corrector.SetBiasFieldFullWidthAtHalfMaximum(
            self.n4_config['fwhm']
        )

        corrected = corrector.Execute(image, mask)
        logger.info("N4 bias field correction completed")
        return corrected

    def normalize_by_tumor_region(
        self,
        image: sitk.Image,
        mask: sitk.Image
    ) -> sitk.Image:
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)

        tumor_values = image_array[mask_array > 0]
        if len(tumor_values) == 0:
            logger.warning("Empty tumor mask, using whole image for normalization")
            tumor_values = image_array.flatten()

        mean_val = np.mean(tumor_values)
        std_val = np.std(tumor_values)

        if std_val < 1e-7:
            logger.warning("Near-zero std in tumor region, skipping normalization")
            return image

        normalized = (image_array - mean_val) / std_val
        clip_low, clip_high = self.norm_config['clip_range']
        normalized = np.clip(normalized, clip_low, clip_high)

        result = sitk.GetImageFromArray(normalized)
        result.CopyInformation(image)
        return result

    def resample_image(self, image: sitk.Image) -> sitk.Image:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(original_size[i] * (original_spacing[i] / self.target_spacing[i])))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetDefaultPixelValue(0)

        return resampler.Execute(image)

    def preprocess_single_sequence(
        self,
        image_path: str,
        mask_path: str,
        output_image_path: Optional[str] = None,
        reference_image_path: Optional[str] = None
    ) -> sitk.Image:
        logger.info(f"Preprocessing: {image_path}")

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        if reference_image_path and os.path.exists(reference_image_path):
            reference = sitk.ReadImage(reference_image_path)
            image = self._resample_to_reference(image, reference)
            mask = self._resample_to_reference(mask, reference, is_mask=True)

        image = self.n4_bias_correction(image)

        image = self.resample_image(image)
        mask = self.resample_mask(mask)

        if self.config['preprocessing']['normalize']:
            image = self.normalize_by_tumor_region(image, mask)

        if output_image_path:
            ensure_dir(os.path.dirname(output_image_path))
            sitk.WriteImage(image, output_image_path)
            logger.info(f"Saved to: {output_image_path}")

        return image

    def _resample_to_reference(
        self,
        image: sitk.Image,
        reference: sitk.Image,
        is_mask: bool = False
    ) -> sitk.Image:
        interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        return sitk.Resample(
            image, reference,
            sitk.Transform(),
            interpolator, 0.0,
            image.GetPixelID()
        )

    def resample_mask(self, mask: sitk.Image) -> sitk.Image:
        original_spacing = mask.GetSpacing()
        original_size = mask.GetSize()

        new_size = [
            int(round(original_size[i] * (original_spacing[i] / self.target_spacing[i])))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputOrigin(mask.GetOrigin())
        resampler.SetOutputDirection(mask.GetDirection())
        resampler.SetDefaultPixelValue(0)

        return resampler.Execute(mask)

    def preprocess_patient(
        self,
        patient_dir: str,
        output_dir: str,
        timepoint: str = "pre"
    ) -> Dict[str, str]:
        logger.info(f"Preprocessing patient: {patient_dir} ({timepoint})")

        ensure_dir(output_dir)
        output_paths = {}
        reference_image = None

        for seq_info in self.sequences:
            seq_code = seq_info['code']
            image_filename = f"{seq_code}.nii.gz"
            image_path = os.path.join(patient_dir, image_filename)

            if not os.path.exists(image_path):
                logger.warning(f"Sequence not found: {image_path}")
                continue

            output_image_path = os.path.join(
                output_dir, f"{timepoint}_{seq_code}.nii.gz"
            )

            mask_filename = f"{seq_code}_mask.nii.gz"
            mask_path = os.path.join(patient_dir, mask_filename)
            if not os.path.exists(mask_path):
                mask_path = os.path.join(patient_dir, "tumor_mask.nii.gz")
            if not os.path.exists(mask_path):
                logger.warning(f"Mask not found for {seq_code}")
                continue

            ref_path = None
            if not seq_info.get('is_reference', False) and reference_image:
                ref_path = reference_image

            image = self.preprocess_single_sequence(
                image_path=image_path,
                mask_path=mask_path,
                output_image_path=output_image_path,
                reference_image_path=ref_path
            )

            output_paths[seq_code] = output_image_path

            if seq_info.get('is_reference', False) and reference_image is None:
                reference_image = output_image_path

        mask_output_path = os.path.join(output_dir, f"{timepoint}_mask.nii.gz")
        if os.path.exists(mask_path):
            mask = sitk.ReadImage(mask_path)
            mask = self.resample_mask(mask)
            sitk.WriteImage(mask, mask_output_path)
            output_paths['mask'] = mask_output_path

        return output_paths

    def batch_preprocess(
        self,
        raw_dir: str,
        output_dir: str,
        timepoint: str = "pre"
    ) -> None:
        logger.info(f"Batch preprocessing: {raw_dir} -> {output_dir}")

        ensure_dir(output_dir)

        patient_dirs = sorted([
            d for d in os.listdir(raw_dir)
            if os.path.isdir(os.path.join(raw_dir, d))
        ])

        logger.info(f"Found {len(patient_dirs)} patients")

        for patient_id in tqdm(patient_dirs, desc=f"Preprocessing ({timepoint})"):
            patient_dir = os.path.join(raw_dir, patient_id)
            patient_output = os.path.join(output_dir, patient_id)

            try:
                self.preprocess_patient(
                    patient_dir=patient_dir,
                    output_dir=patient_output,
                    timepoint=timepoint
                )
            except Exception as e:
                logger.error(f"Error processing {patient_id}: {str(e)}")
                continue

        logger.info(f"Batch preprocessing completed: {len(patient_dirs)} patients")

    def register_pre_to_post(
        self,
        pre_image_path: str,
        post_image_path: str,
        output_path: str
    ) -> sitk.Image:
        return self.register_images(
            fixed_image_path=pre_image_path,
            moving_image_path=post_image_path,
            output_path=output_path
        )
