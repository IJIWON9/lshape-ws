import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Road 안에 있는 Lane들이 잘 정렬되어있나 plot을 통해 확인하는 코드

# Load the JSON data from a file
with open('etc/hd_map_parser/output_file/map_data_detail.json', 'r') as file:
    data = json.load(file)

def animate_multiple_lanes(lane_numbers, interval=100, point_size=2):
    fig, ax = plt.subplots()

    lines = []
    x_coords_list = []
    y_coords_list = []

    # Prepare plots for each lane
    for lane_number in lane_numbers:
        for road in data['roads']:
            for lane in road['lanes']:
                if lane['ID'] == lane_number:
                    x_coords = lane['x']
                    y_coords = lane['y']
                    x_coords_list.append(x_coords)
                    y_coords_list.append(y_coords)
                    line, = ax.plot([], [], marker='o', linestyle='', markersize=point_size, label=f'Lane {lane_number}')
                    lines.append(line)
                    break

    # Determine the x and y limits based on all lanes
    all_x_coords = [x for coords in x_coords_list for x in coords]
    all_y_coords = [y for coords in y_coords_list for y in coords]
    ax.set_xlim(min(all_x_coords) - 10, max(all_x_coords) + 10)
    ax.set_ylim(min(all_y_coords) - 10, max(all_y_coords) + 10)

    def update(num):
        for line, x_coords, y_coords in zip(lines, x_coords_list, y_coords_list):
            line.set_data(x_coords[:num+1], y_coords[:num+1])
        return lines

    ani = animation.FuncAnimation(fig, update, frames=len(max(x_coords_list, key=len)), interval=interval, blit=True, repeat=False)
    plt.title('Multiple Lanes Coordinates Animation')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()

n = 29
# Example usage 31 32 33
lane_numbers = [5*n+1, 5*n+2, 5*n+3, 5*n+4, 5*n+5]  # Replace with the desired lane numbers
animate_multiple_lanes(lane_numbers, interval=100, point_size=4)  # point_size controls the size of the points

