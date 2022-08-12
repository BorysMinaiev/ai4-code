## How to run?

- run server.py from the directory with input files
- in vscode: ctrl + p, build, watch
- open index.html with live-server

## Description

It can show one notebook at a time.
For each notebook, we know the correct order of all cells, and the order, which was predicted by our program. Each cell is represented by one line on the left image. The left end of the line corresponds to the correct position of the cell, right end corresponds to the position predicted by our program. 

Ideally, all lines should be parallel to the bottom of the screen. If it is not parallel our prediction was wrong. You can pick one line (which corresponds to the markdown cell), and it will show corresponding cell (and all code cells) on the right side. For the markdown cell, two positions are shown. The green one is the correct position. Red one - how we predicted it.


![photo_2022-08-12_10-44-11](https://user-images.githubusercontent.com/2011126/184329148-8b256947-8702-4ed9-a32c-27f4e0f4921b.jpg)
