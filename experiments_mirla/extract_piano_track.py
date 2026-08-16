import os
import glob
import mido

SOURCE_FOLDER = "data/Pop1K7"
OUTPUT_FOLDER = "data/Pop1K7_piano_single"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = (
    glob.glob(SOURCE_FOLDER + "/**/*.mid", recursive=True)
    + glob.glob(SOURCE_FOLDER + "/**/*.midi", recursive=True)
)

converted = 0
skipped = 0

for path in files:
    try:
        midi = mido.MidiFile(path)

        if len(midi.tracks) < 2:
            skipped += 1
            continue

        piano_track = midi.tracks[1]

        new_midi = mido.MidiFile(
            ticks_per_beat=midi.ticks_per_beat
        )

        new_track = mido.MidiTrack()

        for msg in piano_track:
            new_track.append(msg.copy())

        new_midi.tracks.append(new_track)

        filename = os.path.basename(path)

        output_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        new_midi.save(output_path)

        converted += 1

    except Exception as e:
        print("ERROR:", path, e)
        skipped += 1


print()
print("TOTAL FILES:", len(files))
print("CONVERTED:", converted)
print("SKIPPED:", skipped)
print("SAVED TO:", OUTPUT_FOLDER)