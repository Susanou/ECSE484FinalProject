#  Copyright (c) 2019, Cameron Hochberg
#  All rights reserved.
#
# Author: Cameron Hochberg
# Date: 06/2019
# Homepage: https://github.com/Susanou
# Email: cam.hochberg@gmail.com
#

from multiprocessing import Pool
from os.path import isfile, join

from utils import Loader

import sys, os
import argparse

def writer(path: str, file: str, stem: str, words: int, files: int):
    """Fonction pour couper les fichiers textes en fichiers de 500 mots chacun
    
    Parameters
    ----------
    path : str
        Chemin pour acceder au fichier
    file : str
        nom du fichier
    """
    try:
        f = open(file, "r", encoding="utf8")     # open the file in reading mode
        if f.mode == 'r':
            i = 1
            word = 0
            lines = f.readlines()

            while word < words and i < files:
                for line in lines:
                    line = line.split(" ")                        

                    for mot in line:
                        if word == 0:
                            w = open(path + "/"+ stem + "-%d.txt"%i, 'w')
                        w.write(mot+" ")
                        word+=1

                        if word == words:
                            w.close()
                            word = 0
                            i += 1

                        if i == files: # This is just to prevent too many files from being generated
                            break

                    if word == words:
                        w.close()
                        word = 0
                        i += 1
                    
                    if i == files:
                        break
            
            w.close()
        
        f.close()
    except IOError:
        print("No file with that name was found\n")
    finally:
        return 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Programm to split all large texts in a folder into smaller paragraphs of fixed length')

    parser.add_argument("pathName", type=str, help="path of the folder with the original files")
    parser.add_argument("newPath", type=str, help="path to the folder where you store the files")
    parser.add_argument("words", type=int, default=100, nargs='?', help="number of words per files")
    parser.add_argument("files", type=int, default=5000, nargs='?', help="number of files to create")

    args = parser.parse_args()

    loader = Loader("Loading...", "All done!", 0.05).start()
    for x in os.listdir(args.pathName):
        #print(args.newPath)
        if isfile(join(args.pathName, x)):
            print(join(args.pathName, x))
            writer(args.newPath, join(args.pathName, x), x.split(".")[0], args.words, args.files)  
    loader.stop()