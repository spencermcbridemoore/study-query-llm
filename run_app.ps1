# Run the Study Query LLM Panel app
$env:PYTHONPATH = $PSScriptRoot
$env:PYTHONNOUSERSITE = "1"
python "$PSScriptRoot\panel_app\app.py" @args
