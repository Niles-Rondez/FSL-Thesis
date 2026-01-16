import os
from pathlib import Path

# Expected structure
EXPECTED_PARTICIPANTS = [f"H{i}" for i in range(1, 11)]
EXPECTED_CLASSES = sorted(
    [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in ['J', 'Z']] + 
    [str(i) for i in range(10)]
)
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def validate_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    errors = []
    warnings = []
    
    # Check root folder exists
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset folder '{dataset_path}' not found!")
        return False
    
    # Check participant folders
    found_participants = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    if found_participants != EXPECTED_PARTICIPANTS:
        errors.append(f"Participant folders mismatch!")
        errors.append(f"  Expected: {EXPECTED_PARTICIPANTS}")
        errors.append(f"  Found: {found_participants}")
    
    # Check each participant
    for participant in EXPECTED_PARTICIPANTS:
        participant_path = dataset_path / participant
        
        if not participant_path.exists():
            errors.append(f"Missing participant folder: {participant}")
            continue
        
        # Check class folders
        found_classes = sorted([d.name for d in participant_path.iterdir() if d.is_dir()])
        
        if found_classes != EXPECTED_CLASSES:
            errors.append(f"{participant}: Class folders mismatch!")
            missing = set(EXPECTED_CLASSES) - set(found_classes)
            extra = set(found_classes) - set(EXPECTED_CLASSES)
            if missing:
                errors.append(f"  Missing classes: {sorted(missing)}")
            if extra:
                errors.append(f"  Extra classes: {sorted(extra)}")
        
        # Check each class folder
        for class_name in EXPECTED_CLASSES:
            class_path = participant_path / class_name
            
            if not class_path.exists():
                continue
            
            # Count images
            images = [f for f in class_path.iterdir() 
                     if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
            
            if len(images) == 0:
                errors.append(f"{participant}/{class_name}: No images found!")
            elif len(images) < 30:
                warnings.append(f"{participant}/{class_name}: Only {len(images)} images (minimum 30 recommended)")
            
            # Check for invalid files
            all_files = [f for f in class_path.iterdir() if f.is_file()]
            invalid = [f for f in all_files if f.suffix.lower() not in ALLOWED_EXTENSIONS]
            if invalid:
                errors.append(f"{participant}/{class_name}: Invalid files found: {[f.name for f in invalid]}")
    
    # Print results
    print("\n" + "="*70)
    print("DATASET VALIDATION REPORT")
    print("="*70)
    
    if not errors and not warnings:
        print("\n✅ SUCCESS! Dataset structure is perfect!")
        print(f"\nDataset location: {dataset_path.absolute()}")
        print(f"Participants: {len(EXPECTED_PARTICIPANTS)}")
        print(f"Classes per participant: {len(EXPECTED_CLASSES)}")
        return True
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print("\n❌ ERRORS (MUST FIX):")
        for e in errors:
            print(f"  - {e}")
        print("\n❌ VALIDATION FAILED - Fix errors before proceeding!")
        return False
    
    print("\n⚠️  VALIDATION PASSED WITH WARNINGS")
    print("Consider fixing warnings for better results.")
    return True

if __name__ == "__main__":
    # Run validation
    result = validate_dataset("fsl_dataset")
    
    if result:
        print("\n✅ Dataset is ready to zip!")
    else:
        print("\n❌ Fix errors and run validation again.")