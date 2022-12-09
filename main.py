import numpy as np
import cv2
import math

class Point:

    def __init__(self, x, y, z, x_index = -1, y_index = -1):
        self.x = x
        self.y = y
        self.z = z
        self.x_index = x_index
        self.y_index = y_index
        self.cluster = -1
        self.min_distance = np.inf

    def __str__(self):
        return f"[{self.x_index}, {self.y_index}, ({self.x}, {self.y}, {self.z})]"

    def as_array(self):
        return np.array([self.x, self.y, self.z])

def get_euclidean(p1, p2):
    return np.sqrt(np.sum((p1.as_array() - p2.as_array())**2))

def kmeans(image, iterations, clusters, initial_centroids = None):

    centroids = np.random.randint(0, 255, size=(clusters, image.shape[2]))
    centroids = [Point(x, y, z) for x, y, z in centroids]
    if initial_centroids is not None: centroids = initial_centroids
    points = []
    
    for i in range(rows):
        for j in range(cols):
            point = Point(image[i][j][0], image[i][j][1], image[i][j][2], i, j)
            points.append(point)

    for _ in range(iterations):

        print(f"KMeans iteration {_}...")

        for k in range(clusters):
            centroid = centroids[k]
            for point in points:
                distance = get_euclidean(centroid, point)
                if distance < point.min_distance:
                    point.min_distance = distance
                    point.cluster = k

        total = np.zeros(clusters, dtype=int)
        sum_centroids = np.zeros((clusters, image.shape[2]), dtype=int)

        for point in points:
            cluster = point.cluster
            total[cluster] += 1
            sum_centroids[cluster] += point.as_array()

        for i in range(clusters):
            if total[i] == 0: total[i] = 1
            x, y, z = sum_centroids[i] / total[i]
            centroids[i] = Point(x, y, z)

    return points, centroids

def binarize_rg(image, threshold = 40):

    rows, cols = image.shape[:2]
    result = np.zeros((rows, cols), np.uint8)

    for i in range(rows):
        for j in range(cols):

            b = image[i][j][0]
            if b < threshold: 
                result[i][j] = 0
            else:
                result[i][j] = 255

    cv2.imwrite("binarized.jpg", result)

    return result

def inverse(image):
    rows, cols = image.shape[:2]
    for i in range(rows):
        for j in range(cols):
            if image[i][j] == 0: image[i][j] = 255
            else: image[i][j] = 0

def object_tagging(image):
    
    rows, cols = image.shape[:2]
    result = image.copy()
    result_border = np.zeros((rows, cols), np.uint8)
    tag = 1
    discard_x = rows / 2
    discard_y = cols / 2

    result_mask = image.copy()
    cv2.floodFill(result_mask, None, (0, 0), 0)
    inverse(result_mask)
    cv2.imwrite("inverse.jpg", result_mask)

    for i in range(rows):
        for j in range(cols):
            if result_mask[i][j] != result[i][j]:
                result[i][j] = 0

    cv2.imwrite("pre_tagging.jpg", result)

    for i in range(rows):
        for j in range(cols):
            if result[i][j] == 0:
                cv2.floodFill(result, None, (j, i), tag)
                tag += 1

    # result = np.zeros((rows, cols), np.uint8)
    # result_border = np.zeros((rows, cols), np.uint8)

    # active = False
    # tag = 1

    # for i in range(rows):
    #     for j in range(cols):
    #         if i < discard_x and j < discard_y: continue
    #         if image[i][j] == 0:
    #             if active:
    #                 result[i][j] = tag
    #             else:
    #                 active = True
    #                 tag += 1
    #                 result[i][j] = tag
    #         else:
    #             active = False

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            if i < discard_x and j < discard_y: continue
            if result[i][j] == 0: continue

            is_border = False
            
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if result[i, j] != result[i + x][j + y]:
                        is_border = True

            if is_border: result_border[i][j] = result[i][j]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if result_border[i][j] != 0 and result_border[i + x][j + y] != 0 and result_border[i][j] != result_border[i + x][j + y]:
                        result_border[i, j] = min(result_border[i, j], result_border[i + x][j + y])
                        result_border[i + x][j + y] = result_border[i][j]

    tags_dict = {}
    for i in range(rows):
        for j in range(cols):
            if result_border[i][j] == 0: continue
            if result_border[i][j] in tags_dict:
                tags_dict[result_border[i][j]]["points"].append(Point(i, j, 1))
                tags_dict[result_border[i][j]]["count"] += 1
            else:
                tags_dict[result_border[i][j]] = {
                    "points": [Point(i, j, 1)],
                    "count": 1
                }

    filtered_tags = {}
    for tag in tags_dict:
        if tags_dict[tag]["count"] < 2000: continue
        filtered_tags[tag] = tags_dict[tag]

    color_tag = np.zeros((rows, cols, 3), np.uint8)
    for tag in filtered_tags:
        print("Tag: ", tag, "Count: ", filtered_tags[tag]["count"])
        points = filtered_tags[tag]["points"]
        for point in points:
            color_tag[point.x][point.y] = [(20 * tag) + 20, 255 - (20 * tag), 255 - (20 * tag)]

    cv2.imwrite("color_tag.jpg", color_tag)
        
    cv2.imwrite("tagging_border_disc.jpg", result_border)

    return result_border, filtered_tags

def find_farthests(tags):

    farthests = []
    for idx, tag in enumerate(tags):

        points = tags[tag]["points"]
        distance = 0
        farthest_1 = None
        farthest_2 = None

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                temp_distance = get_euclidean(points[i], points[j])
                if temp_distance > distance:
                    distance = temp_distance
                    farthest_1 = points[i]
                    farthest_2 = points[j]

        if farthest_1 is not None and farthest_2 is not None:
            farthests.append((farthest_1, farthest_2, tag))

    return farthests

def cross_product(u, v):

    u_x = u[0]
    u_y = u[1]
    u_z = u[2]

    v_x = v[0]
    v_y = v[1]
    v_z = v[2]

    w_x = (u_y * v_z) - (u_z * v_y)
    w_y = (u_z * v_x) - (u_x * v_z)
    w_z = (u_x * v_y) - (u_y * v_x)

    if w_z == 0: w_z = 1

    wx = w_x / w_z
    wy = w_y / w_z
    wz = w_z / w_z

    return Point(wx, wy, wz)

def get_line_str(u):
    return "y = " + str((-u[0] / u[1])) + "x " + str(-u[2] / u[1])

def get_line_points(u, p1, p2):

    x1 = p1.y
    x2 = p2.y
    points = []

    if x1 > x2: x1, x2 = x2, x1
    points = [ (math.floor((-u[2] - u[0] * x) / u[1]), x) for x in range(x1, x2 + 1) ]

    return points

if __name__ == "__main__":
    
    image = cv2.imread("Jit1.JPG", cv2.IMREAD_COLOR)
    rows, cols = image.shape[:2]

    # KMeans

    print("Starting KMeans...")
    clusters = 5
    iterations = 5
    kmeans_image = np.zeros(image.shape, dtype=np.uint8)
    initial_centroids = np.array([
        Point(0, 0, 0),
        Point(0, 0, 255),
        Point(0, 255, 0),
        Point(255, 0, 0),
        Point(255, 255, 255),
    ])

    points, centroids = kmeans(image, iterations, clusters, initial_centroids)
    centroids = np.array([np.round(centroid.as_array()) for centroid in centroids])
    for point in points: kmeans_image[point.x_index][point.y_index] = centroids[point.cluster]
    cv2.imwrite("kmeans.jpg", kmeans_image)

    # Binarization
    print("Binarization...")
    binarized = binarize_rg(kmeans_image)

    # Tagging
    print("Tagging...")
    tagged, tags = object_tagging(binarized)
    farthests_points = find_farthests(tags)

    # Find centers
    print("Finding lines...")
    for points in farthests_points:
        
        pm = Point((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2, 1)
        pm.x = math.floor(pm.x)
        pm.y = math.floor(pm.y)

        # Obj 2
        if pm.x < rows / 2: 

            i_l = pm.y
            i_r = pm.y

            f_il = pm.y
            f_ir = pm.y

            while i_l > 0:
                if tagged[pm.x][i_l] == points[2]: f_il = i_l
                i_l -= 1
            while i_r < cols: 
                if tagged[pm.x][i_r] == points[2]: f_ir = i_r
                i_r += 1

            left = Point(pm.x, f_il, 1)
            right = Point(pm.x, f_ir, 1)
            line = cross_product([
                left.y,
                left.x,
                1
            ], [
                right.y,
                right.x,
                1
            ])

            print(f"Obj 2: ({left.y}, {left.x})) | ({right.y}, {right.x}), Distance: {get_euclidean(left, right)}")
            print(f"Line: {line.x} {line.y} {line.z}")
            print(get_line_str([line.x, line.y, line.z]))
            points = get_line_points([line.x, line.y, line.z], left, right)

            for point in points:
                image[point[0]][point[1]] = (0, 255, 0)

            # cv2.line(image, (left.y, left.x), (right.y, right.x), (0, 255, 0), 1)

        # Obj 4
        else:

            i_l = pm.y
            j_l = pm.x
            f_il = pm.y
            f_jl = pm.x

            i_r = pm.y
            j_r = pm.x
            f_ir = pm.y
            f_jr = pm.x

            while i_l > 0 and j_l < rows:
                if tagged[j_l][i_l] == points[2]:
                    f_il = i_l
                    f_jl = j_l
                i_l -= 1
                j_l += 1

            while i_r < cols and j_r > 0:
                if tagged[j_r][i_r] == points[2]:
                    f_ir = i_r
                    f_jr = j_r
                i_r += 1
                j_r -= 1

            left = Point(f_jl, f_il, 1)
            right = Point(f_jr, f_ir, 1)
            line = cross_product([
                left.y,
                left.x,
                1
            ], [
                right.y,
                right.x,
                1
            ])

            print(f"Obj 4: ({left.y}, {left.x})) | ({right.y}, {right.x}), Distance: {get_euclidean(left, right)}")
            print(f"Line: {line.x} {line.y} {line.z}")
            print(get_line_str([line.x, line.y, line.z]))
            points = get_line_points([line.x, line.y, line.z], left, right)
            
            for point in points:
                image[point[0]][point[1]] = (0, 255, 0)
            # cv2.line(image, (left.y, left.x), (right.y, right.x), (0, 255, 0), 1)

    cv2.imwrite("output.jpg", image)

    # cv2.imshow("Original", image)
    # cv2.imshow("KMeans", kmeans_image)
    # cv2.imshow("binarized", binarized)
    # cv2.imshow("Tagged", tagged)

    cv2.waitKey(0)