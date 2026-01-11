import os
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
import config

class SubjectAwareDataset:
    """Manages subject-aware data loading for LOSO evaluation"""
    
    def __init__(self, data_root: str = config.DATA_ROOT):
        self.data_root = Path(data_root)
        self.subjects = self._discover_subjects()
        self.class_to_idx = {cls: i for i, cls in enumerate(config.CLASSES)}
        
    def _discover_subjects(self) -> List[str]:
        """Find all subject folders matching pattern"""
        subjects = sorted([d.name for d in self.data_root.glob(config.SUBJECT_PATTERN) 
                          if d.is_dir()])
        if not subjects:
            raise ValueError(f"No subjects found in {self.data_root} matching {config.SUBJECT_PATTERN}")
        return subjects
    
    def get_subject_data(self, subjects: List[str], transform=None) -> datasets.ImageFolder:
        """Create ImageFolder dataset from specified subjects"""
        # Create temporary combined directory structure
        # OR: Use custom Dataset class that filters by subject
        # Implementation: Custom dataset that checks parent folder
        
        from torch.utils.data import ConcatDataset
        datasets_list = []
        for subj in subjects:
            subj_path = self.data_root / subj
            if not subj_path.exists():
                raise ValueError(f"Subject folder not found: {subj_path}")
            dataset = datasets.ImageFolder(subj_path, transform=transform)
            datasets_list.append(dataset)
        
        return ConcatDataset(datasets_list)
    
    def get_loso_splits(self, fold: int) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
        """Return train/test datasets for LOSO fold"""
        if fold < 0 or fold >= len(self.subjects):
            raise ValueError(f"Fold {fold} out of range [0, {len(self.subjects)-1}]")
        
        test_subject = [self.subjects[fold]]
        train_subjects = [s for s in self.subjects if s != test_subject[0]]
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
        ])
        
        train_data = self.get_subject_data(train_subjects, train_transform)
        test_data = self.get_subject_data(test_subject, test_transform)
        
        return train_data, test_data, test_subject[0]
    
    def get_all_subjects_data(self) -> datasets.ImageFolder:
        """Return dataset with all subjects for final training"""
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
        ])
        return self.get_subject_data(self.subjects, transform)