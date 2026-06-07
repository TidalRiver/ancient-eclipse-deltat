# ancient-eclipse-deltat

Python reproduction of ancient total eclipse constraints on Delta T and Earth's long-term rotational slowdown.

This project reproduces a method that uses historical total solar eclipse records to constrain the possible range of Delta T.

## Files

- `exo-deltaT.py`: main calculation script. It scans Delta T values for a selected eclipse date and observing site.
- `exo-deltaTFigure.py`: plotting script. It reads result files and generates eclipse figures or maps.

## External data

Large ephemeris/data files are not included in this repository.

In particular, the SPICE kernel file, such as:

- de441_part-1.bsp

should be prepared locally by the user.

If the file is stored in a local files/ folder, the expected structure can be:

    ancient-eclipse-deltat/
    ├── exo-deltaT.py
    ├── exo-deltaTFigure.py
    └── files/
        └── de441_part-1.bsp

The files/ folder is ignored by Git because the ephemeris file can be very large.

On macOS, if the data file is stored on an external drive, the path usually starts with:

    /Volumes/

For example:

    /Volumes/YOUR_DISK_NAME/path/to/de441_part-1.bsp

## Example run

A typical command for `exo-deltaT.py` looks like:

    python exo-deltaT.py \
      --kernel "files/de441_part-1.bsp" \
      --year 1542 \
      --month 8 \
      --day 11 \
      --lat "35 36 00N" \
      --lon "116 59 00E" \
      --delta-t-min -1000 \
      --delta-t-max 1000 \
      --calendar julian \
      --output result.txt

Please modify the date, site coordinates, Delta T range, and kernel path according to the eclipse case being studied.

## Notes for `exo-deltaTFigure.py`

Some parameters in `exo-deltaTFigure.py` may need to be edited manually before running, including:

- the result files to be read
- the SPICE kernel path
- the eclipse date
- the Delta T values to be plotted
- the map longitude and latitude range
- the marked observing sites

This code is intended as a reproduction and exploratory implementation of a published method, not as a fully packaged command-line tool.
