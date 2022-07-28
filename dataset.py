from sklearn.model_selection import train_test_split
import scipy.sparse as sparse

my_checkin_path = "./data/mydata/checkins/"
my_coords_path = "./data/mydata/coords/"
city = "Kyoto"
persent = "0.8"

class Foursquare(object):
    def __init__(self):
        self.user_num = 783
        self.poi_num = 1496

    def read_raw_data(self):
        directory_path = my_checkin_path
        checkin_file = city + '_checkins.txt'
        all_data = open(directory_path + checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid = int(uid), int(lid)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1

        return sparse_raw_matrix.tocsr()
    
    def read_train_data(self):
        directory_path = my_checkin_path
        checkin_file = city + '_' + persent + '_checkins_train_2.txt'
        all_data = open(directory_path + checkin_file, 'r').readlines()
        sparse_train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid = int(uid), int(lid)
            sparse_train_matrix[uid, lid] = sparse_train_matrix[uid, lid] + 1

        return sparse_train_matrix.tocsr()
    
    def read_test_set(self):
        directory_path = my_checkin_path
        checkin_file = city + '_' + persent + '_checkins_test_2.txt'
        all_data = open(directory_path + checkin_file, 'r').readlines()
        # sparse_train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        test_place = []
        for i in range(self.user_num):
            test_place.append([])

        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            test_place[int(uid)].append(int(lid))
            # uid, lid = int(uid), int(lid)
            # sparse_train_matrix[uid, lid] = sparse_train_matrix[uid, lid] + 1

        return test_place

    def split_data(self, raw_matrix, random_seed=0):
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        test_set = []
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            
            try:
                train_place, test_place, train_freq, test_freq = train_test_split(place_list, freq_list, test_size=0.2, random_state=random_seed)
            except ValueError:
                # print("train size is 0")
                train_place, test_place, train_freq, test_freq = train_test_split(place_list, freq_list, train_size=1, random_state=random_seed)
            # print(test_place)
            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]
            test_set.append(test_place.tolist())

        return train_matrix.tocsr(), test_set

    def read_poi_coos(self):
        directory_path = my_coords_path
        poi_file = city+'_poi_coos.txt'
        poi_coos = {}
        poi_data = open(directory_path + poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])

        return place_coords

    def generate_data(self, random_seed=0):
        raw_matrix = self.read_raw_data()
        # train_matrix, test_set = self.split_data(raw_matrix, random_seed)
        train_matrix = self.read_train_data()
        test_set = self.read_test_set()
        place_coords =self.read_poi_coos()
        return train_matrix, test_set, place_coords


if __name__ == '__main__':
    train_matrix, test_set, place_coords = Foursquare().generate_data()
    print(train_matrix.shape, len(test_set), len(place_coords))


