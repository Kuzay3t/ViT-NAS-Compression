# src/baselines/download_baselines.py
"""
Download and organize baseline ViT models and NAS frameworks.
"""
import os
import sys
from pathlib import Path
from typing import Dict
import subprocess

class BaselineDownloader:
    """Download baseline models and frameworks."""
    
    def __init__(self, base_dir: str = "baselines"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def clone_repo(url: str, target_dir: Path, name: str):
        """Clone a GitHub repository."""
        target_dir.mkdir(parents=True, exist_ok=True)
        repo_path = target_dir / name
        
        if repo_path.exists():
            print(f"Repository {name} already exists at {repo_path}")
            return repo_path
        
        try:
            print(f"Cloning {name} from {url}...")
            subprocess.run(
                ["git", "clone", url, str(repo_path)],
                check=True,
                capture_output=True
            )
            print(f"Successfully cloned {name}")
            return repo_path
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {name}: {e}")
            return None
    
    def download_all_baselines(self):
        """Download all baseline repositories."""
        
        baselines = {
            "ProxylessNAS": "https://github.com/mit-han-lab/proxylessnas.git",
            "HAQ": "https://github.com/mit-han-lab/haq.git",
            "APQ": "https://github.com/YuhuaZhu/APQ.git",
            "DynamicViT": "https://github.com/blackfeather-wang/DynamicViT.git",
            "ViT": "https://github.com/google-research/vision_transformer.git",
            "timm": "https://github.com/rwightman/pytorch-image-models.git",
        }
        
        results = {}
        for name, url in baselines.items():
            repo_path = self.clone_repo(url, self.base_dir, name)
            results[name] = str(repo_path) if repo_path else None
        
        return results
    
    def setup_pythonpath(self):
        """Add baseline directories to PYTHONPATH."""
        for baseline_dir in self.base_dir.iterdir():
            if baseline_dir.is_dir():
                sys.path.insert(0, str(baseline_dir))
                print(f"Added {baseline_dir} to PYTHONPATH")

def main():
    """Main entry point."""
    print("Starting baseline downloads...")
    print("This may take a while (~5-10 minutes)...\n")
    
    downloader = BaselineDownloader()
    
    results = downloader.download_all_baselines()
    downloader.setup_pythonpath()
    
    print("\n" + "="*60)
    print("BASELINE DOWNLOAD SUMMARY")
    print("="*60)
    for name, path in results.items():
        status = "✓ Success" if path else "✗ Failed"
        print(f"{name:.<30} {status}")
        if path:
            print(f"  Location: {path}")
    print("="*60)

if __name__ == "__main__":
    main()
