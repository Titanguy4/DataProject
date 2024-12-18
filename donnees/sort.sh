#!/bin/bash

experience_size=601
directories=($(ls -d */ | grep -v '^experience'))

for value in "${directories[@]}"
do
  if (( $(ls $value | wc -l) != 8414 ))
  then
    echo "Oh cousin ti'a chie dans la colle, ya pas le meme nb de fichier de partout !"
    exit
  fi
  nb_experiences=$(($(ls $value | wc -l)/$experience_size))
done

out_directory='experiences'
mkdir $out_directory

for i in $(seq 1 $nb_experiences);
do
  if [ -d "${i}" ]
  then
    echo "dossier deja cree"
  else
    mkdir "${out_directory}/${i}"
    for value in "${directories[@]}"
    do
      mkdir "${out_directory}/${i}/$value"
      start=$(($experience_size*$(($i-1))))
      end=$(($(($experience_size*$i))-1))
      prefix="${value%_*}_"
      for j in $(seq $start $end);
      do
        cp "${value}/${prefix}${j}.txt" "${out_directory}/${i}/$value/"
      	echo "Création fichier..."
	done
    done
  fi
done
