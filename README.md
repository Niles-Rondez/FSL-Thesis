# FSL-Thesis
# Pls do these thank you very much uwu

# activate env
& .venv\Scripts\Activate.ps1

# 1. check dataset
python check_dataset.py

# 2. split
run python make_splits.py

# 3. train resnet50
run python train_resnet_fsl.py

# 4. evaluate
run python evaluate_model.py

# 5. benchmark
run python benchmark_inference.py

# 6. validator review
run python validator_tool.py

# 7. run prototype UI
set FLASK_APP=app.py
flask run
