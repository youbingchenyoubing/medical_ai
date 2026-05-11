
import requests

TCIA_API_BASE = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"

def test_tcia_connection():
    print("Testing TCIA API connection...")
    try:
        # 测试获取可用的数据集列表
        url = f"{TCIA_API_BASE}/getCollectionValues"
        print(f"Requesting: {url}")
        
        response = requests.get(url, timeout=30)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            collections = response.json()
            print(f"Successfully retrieved {len(collections)} collections!")
            print("First 5 collections:", collections[:5])
            return True
        else:
            print(f"API returned unexpected status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"Request timeout: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_tcia_connection()
    if not success:
        print("\nTCIA API seems to be unreachable.")
        print("\nAlternative download methods:")
        print("1. Visit the official TCIA website to download data manually")
        print("2. Use NBIA Data Retriever tool: https://www.cancerimagingarchive.net/access-data/")

