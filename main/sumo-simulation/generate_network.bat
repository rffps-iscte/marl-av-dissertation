@echo off
REM Run this from the main/sumo-simulation/ folder to regenerate the grid network
REM Requires SUMO to be installed and SUMO_HOME set

netgenerate --grid --grid.number 5 --grid.length 180 ^
  --default.lanenumber 1 --default.speed 13.9 ^
  --tls.guess false ^
  --output-file my_grid5x5.net.xml

echo Grid network generated: my_grid5x5.net.xml
