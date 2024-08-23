import sys
from core import correctImage
from video import analyze_video, process_video

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage")
        print("-" * 20)
        print("For image:")
        print("$python correct.py image <source_image_path> <output_image_path>\n")
        print("-" * 20)
        print("For video:")
        print("$python correct.py video <source_video_path> <output_video_path>\n")
        exit(0)

    if (sys.argv[1]) == "image":
        correctImage(sys.argv[2], sys.argv[3])
    else:
        for item in analyze_video(sys.argv[2], sys.argv[3]):
            if type(item) == dict:
                video_data = item
        [x for x in process_video(video_data, yield_preview=False)]
