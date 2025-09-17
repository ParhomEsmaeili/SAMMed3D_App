docker container run --gpus 1 --rm -ti \
--volume /home/psmeili/IS-Validation-Framework/IS_Validate:/home/psmeili/IS_Validate \
--volume /data/psmeili/Validation_Framework_Datasets/datasets:/home/psmeili/external_mount/datasets \
--volume /data/psmeili/IS_Applications/SAMMed3D_Validate_App/:/home/psmeili/external_mount/input_application/Sample_SAMMed3D \
--volume /data/psmeili/Validation_Results/:/home/psmeili/external_mount/results \
--volume /home/psmeili/IS-Validation-bashscripts:/home/psmeili/validation_bashscripts \
--cpus 10 \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--ipc host \
--name sammed3dv1_test \
testing:sammed3dv1

