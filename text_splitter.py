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
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep

import sys, os
import argparse

class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["[■□□□□□□□□□]","[■■□□□□□□□□]", "[■■■□□□□□□□]", "[■■■■□□□□□□]", "[■■■■■□□□□□]", "[■■■■■■□□□□]", "[■■■■■■■□□□]", "[■■■■■■■■□□]", "[■■■■■■■■■□]", "[■■■■■■■■■■]"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()

def writer(path: str, file: str, words: int, files: int):
    """Fonction pour couper les fichiers textes en fichiers de 500 mots chacun
    
    Parameters
    ----------
    path : str
        Chemin pour acceder au fichier
    file : str
        nom du fichier
    """
    try:
        f = open(file, "r")     # open the file in reading mode
        if f.mode == 'r':
            i = 1
            word = 0
            lines = f.readlines()

            while word < words and i < files:
                for line in lines:
                    line = line.split(" ")                        

                    for mot in line:
                        if word == 0:
                            new = file.split(".")
                            print(path+new[0]+"-%d.txt"%i)
                            w = open(path+new[0]+"-%d.txt"%i, 'w')
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
        if isfile(join(args.pathName, x)):
            writer(args.newPath, join(args.pathName, x), args.words, args.files)  
    loader.stop()