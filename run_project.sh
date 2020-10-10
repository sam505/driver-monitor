source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

python src/main.py --input_type cam --input_file /home/sammie/Downloads/Computer_pointer_controller/bin/demo.mp4 --show_results yes --model_path models/intel/

