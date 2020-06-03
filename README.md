# Code voor masterproef
Deze repository bevat de code voor het project van mijn masterproef omtrent VideoBERT.

## Stap 1: Verzameling van de trainingsdata
In deze stap worden de videos en tekstannotaties verzameld uit de HowTo100M dataset. Het bestand ***stap1/ids.txt*** bevat alle ids van de 47470 videos die opgenomen werden in de trainingsdata.

## Stap 2: Transformatie van de data
In deze stap worden de videos getransformeerd door de frame rate aan te passen naar 10 fps en aan de tekst interpunctie toe te voegen.

## Stap 3: Extractie van de I3D kenmerken
De I3D kenmerken van de videos worden in deze stap geconstrueerd a.d.h.v. het I3D netwerk. De folder ***stap3/checkpoint*** bevat het originele Tensorflow checkpoint voor het I3D model.

## Stap 4: Clustering van de I3D kenmerken
In deze stap worden de I3D kenmerken gegroeppeerd a.d.h.v. hïerarchische k-means. De beste resultaten werden bekomen wanneer ***k=12*** en ***h=4***.

## Stap 5: BERT omvormen tot VideoBERT
Het startpunt van VideoBERT is het BERT model. De ***state_dict*** van het getrainde BERT model kan in ***Pytorch*** aangepast worden om rekening te houden met de nieuwe woordenschat.

## Stap 6: Training van het model
In de laatste stap werd het model getraind.

# Bronnen
De belangrijkste bronnen zijn:

  - HuggingFace: https://github.com/huggingface/transformers
  - I3D model: https://github.com/deepmind/kinetics-i3d
  - RevoScaleR: https://docs.microsoft.com/en-us/machine-learning-server/r-reference/revoscaler/revoscaler
  - FFmpeg: https://ffmpeg.org/
  - Youtube-dl: https://youtube-dl.org/
  - HowTo100M data: https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/
  - YouCookII: http://youcook2.eecs.umich.edu/
