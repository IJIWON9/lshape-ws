import json

# Lane별로 시작 위치가 다르면(거꾸로 되어있으면) Lane을 반대로 정렬함

# Load the JSON data from a file
with open('etc/hd_map_parser/output_file/map_data_detail.json', 'r') as file:
    data = json.load(file)

def reverse_lane_coordinates(lane_id):
    # Iterate through the roads and lanes to find the matching lane
    for road in data['roads']:
        for lane in road['lanes']:
            if lane['ID'] == lane_id:
                # Reverse the x and y coordinates
                lane['x'] = lane['x'][::-1]
                lane['y'] = lane['y'][::-1]
                
                # Optionally save the modified JSON back to a file
                with open('etc/hd_map_parser/output_file/map_data_detail.json', 'w') as outfile:
                    json.dump(data, outfile, indent=4)
                
                print(f"Lane {lane_id} coordinates have been reversed and saved.")
                return
    
    print(f"Lane {lane_id} not found.")

# Example usage
lane_id = 146  # Replace with the desired lane ID
reverse_lane_coordinates(lane_id)
