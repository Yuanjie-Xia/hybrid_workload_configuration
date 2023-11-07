from pymediainfo import MediaInfo
import os
import csv


def find_files(root_dir):
    file_paths = []
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            file_paths.append(os.path.join(foldername, filename))
    return file_paths


def get_workload_feature(media_file_path):
    # Create a MediaInfo object
    media_info = MediaInfo.parse(media_file_path)

    # Access metadata for different tracks in the media file
    feature_list = []
    for track in media_info.tracks:
        if track.track_type == 'Audio':
            # print(f"Track: {track.track_type}")
            duration = track.duration
            Bitrate = track.bit_rate
            Codec = track.codec_id
            feature_list = [duration, Bitrate, Codec]

    return feature_list


def main():
    root_path = 'music_encorder_test_set'
    all_files = find_files(root_path)
    all_files.remove('music_encorder_test_set/.DS_Store')
    workload_feature_set = []
    for file_addr in all_files:
        print(file_addr)
        ext = file_addr.split('.')[-1]
        feature_list = get_workload_feature(file_addr)
        feature_list.append(ext)
        print(feature_list)
        workload_feature_set.append(feature_list)
    csv_file = "workload.csv"
    # Write the list to a CSV file
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in workload_feature_set:
            writer.writerow(row)


if __name__ == "__main__":
    main()