Select (and configure) the inference for the video of your choice via its config file
Ex: ./config_trim4.yaml is for the video trim4.mp4

To run PET with dynamic zoning
python .\PET_deepSORT.py --config .\config_trim4.yaml
To run PET analysis with a single zone of interest:
python .\PET_deepSORT.py --config .\config_trim4.yaml --no-grid --grid-size 8


To run just deepSORT inference run 
python ./deepSORT_rtdetr.py --config ./config_trim2.yaml 

use --deadzone to include non-inference zones