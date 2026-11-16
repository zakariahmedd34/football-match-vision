import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_distance_between
from utils.logger import log_step

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70 #max distance 70 pixels otherwis the ball no assign to anyone

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player["bbox"]
            bottom_left_foot = (player_bbox[0],player_bbox[-1])
            distance_left = get_distance_between(bottom_left_foot, ball_position)            
            bottom_right_foot = (player_bbox[2],player_bbox[-1])
            distance_right = get_distance_between(bottom_right_foot, ball_position)            
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                # Keep the closest player found so far.
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player

    def assign_ball_for_batch(self, player_tracks, ball_tracks, team_ball_control):
        """For each frame of a batch: mark the player with the ball ('has_ball') and append
        his team to team_ball_control (the whole-video list, changed in place).
        No-one near the ball -> the last team keeps it (0 = nobody yet)."""
        with log_step("  Assign ball to player") as out:
            frames_with_owner = 0
            for frame_num, player_track in enumerate(player_tracks):
                ball_bbox = ball_tracks[frame_num][1]['bbox']
                assigned_player = self.assign_ball_to_player(player_track, ball_bbox)
                if assigned_player != -1:
                    frames_with_owner += 1
                    player_track[assigned_player]['has_ball'] = True
                    team_ball_control.append(player_track[assigned_player]['team'])
                else:
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

            out["frames_with_ball_owner"] = f"{frames_with_owner}/{len(player_tracks)}"   # this batch only
            out["control_so_far(team1,team2,none)"] = (team_ball_control.count(1),
                                                       team_ball_control.count(2),
                                                       team_ball_control.count(0))