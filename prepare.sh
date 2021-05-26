
echo ""
echo "########################################"
echo "Download models, data"
echo "########################################"
echo ""

wget  http://data.vision.ee.ethz.ch/alugmayr/SRFlow/datasets.zip
unzip datasets.zip
rm datasets.zip

wget  http://data.vision.ee.ethz.ch/alugmayr/SRFlow/pretrained_models.zip
unzip pretrained_models.zip
rm pretrained_models.zip
