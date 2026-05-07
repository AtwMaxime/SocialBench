# Video sampling configuration used by all builders (except VideoAttentionTarget).
#
# VIDEO_MODE:
#   "framerate"    — downsample to TARGET_FPS frames per second
#   "fixed_number" — extract exactly FIXED_FRAMES frames evenly across the full duration
#                    (output fps = FIXED_FRAMES / duration, variable per clip)

VIDEO_MODE   = "fixed_number"
TARGET_FPS   = 16
FIXED_FRAMES = 16

MERGE_TEST = False  # set to True to produce parquets/benchmark_test.parquet after building
