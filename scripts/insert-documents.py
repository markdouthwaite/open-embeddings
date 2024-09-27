import requests
from typing import Dict
import fire


def main(path: str, base_url: str, **meta: Dict):
    with open(path) as target_file:
        content = target_file.read()
        response = requests.post(
            f"{base_url}/api/v1/documents", json=dict(content=content, **meta)
        )
        if response.ok:
            print("done")
        else:
            print("error:", response.status_code, response.content)


if __name__ == "__main__":
    fire.Fire(main)
