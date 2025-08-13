# CoralSCOP

To run CoralSCOP using Docker, use the following command:

```bash
docker run --rm -it --name coralscope -m 6g -v ./data:/app/data -v dxbarradas/coralscop_custom 

```

This mounts your local `./data` directory to the container's `/app/data` folder.

Inside of the container run :

```bash

cd src/
python segmentation_only_script.py ../data none

```