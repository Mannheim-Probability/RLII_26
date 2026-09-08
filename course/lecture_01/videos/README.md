# Lecture 01 videos

The introductory video `generated/training_progress/training.mp4` is included
in the repository and displayed by the notebook. No training or video-generation
tools are needed to watch it. Other generated clips are not versioned.

## Optional regeneration for instructors

The scripts below are instructor utilities and may not be included in the student
release. Regeneration also requires the original model checkpoints.

Generate the videos from the repository root:

```bash
course/lecture_01/scripts/create_lunarlander_videos.sh
```

The generated files are written to `videos/generated/`:

- `random_lunarlander.mp4`: one seeded episode using uniformly random actions
- `trained_lunarlander.mp4`: 500 steps of the best saved PPO model
- `training_progress/training.mp4`: the random-policy episode followed by a
  labeled montage of the Zoo checkpoint videos

Generated MP4 files are ignored by Git except for the included introductory video. Individual model clips are created with
the Zoo's existing `rl_zoo3.record_video` command. A small course-local helper
adds labels and concatenates them because some ffmpeg installations do not ship
the `drawtext` filter required by the upstream `record_training` command.

The default input is the experiment in
`logs/test_run_for_videos/ppo/LunarLander-v3_1`. Override it if necessary:

```bash
RUN_FOLDER=logs/another_run EXP_ID=2 \
  course/lecture_01/scripts/create_lunarlander_videos.sh
```

`VIDEO_LENGTH` controls the length of each trained-model segment. The Zoo
recorder measures this in environment steps, not seconds. For example:

```bash
VIDEO_LENGTH=400 course/lecture_01/scripts/create_lunarlander_videos.sh
```
