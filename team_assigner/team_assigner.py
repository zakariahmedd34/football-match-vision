from sklearn.cluster import KMeans
from utils.logger import log_step


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}


    def get_clustering_model(self,top_half):
        #reshaoe the image into 2d arrays
        image_2d = top_half.reshape(-1, 3)
        #perform k-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2,init = "k-means++",n_init=1,random_state= 43)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half = image[0:int(image.shape[0]/2), :]
        kmeans = self.get_clustering_model(top_half)
        #get a cluster labels
        labels = kmeans.labels_
        #reshape the labels to the image shape
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])
        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1],clustered_image[-1, 0],clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster 

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        with log_step("  Fit team colors (KMeans, once)", players_in_frame=len(player_detections)) as out:
            player_colors = []
            for _, player_detection in player_detections.items():
                bbox = player_detection["bbox"]
                player_color = self.get_player_color(frame,bbox)
                player_colors.append(player_color)

            kmeans = KMeans(n_clusters=2,init="k-means++", n_init = 10,random_state=43)
            kmeans.fit(player_colors)

            self.kmeans = kmeans

            self.team_colors[1] = kmeans.cluster_centers_[0]
            self.team_colors[2] = kmeans.cluster_centers_[1]

            out["team_colors"] = {t: tuple(int(c) for c in col) for t, col in self.team_colors.items()}

    def assign_teams(self, frames, player_tracks):
        """Add 'team' and 'team_color' to every player box of a batch (changes player_tracks in place)."""
        with log_step("  Assign teams") as out:
            team_counts = {}
            for frame_num, player_track in enumerate(player_tracks):
                for player_id, track in player_track.items():
                    team = self.get_player_team(frames[frame_num], track['bbox'], player_id)
                    track['team'] = team
                    track['team_color'] = self.team_colors[team]
                    team_counts[int(team)] = team_counts.get(int(team), 0) + 1
            out["player_boxes_per_team"] = team_counts

    def get_player_team(self, frame, player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id