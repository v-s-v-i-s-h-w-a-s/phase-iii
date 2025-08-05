import pandas as pd

# Load and analyze the Hawks vs Knicks tracking data
df = pd.read_csv('video_analysis_hawks_vs_knicks/tracking_data.csv')

print("🏀 Hawks vs Knicks - Basketball GNN Analysis Results")
print("=" * 55)
print()

print("📊 Detection Statistics:")
print(f"• Total player detections: {len(df):,}")
print(f"• Unique players identified: {df['player_id'].nunique()}")
print(f"• Frames processed: {df['frame'].nunique()}")
print(f"• Average players per frame: {len(df) / df['frame'].nunique():.1f}")
print(f"• Detection confidence range: {df['confidence'].min():.2f} - {df['confidence'].max():.2f}")
print()

print("👥 Team Classification:")
team_counts = df['team'].value_counts()
print(f"• Team 0 (likely Hawks): {team_counts.get(0, 0)} detections")
print(f"• Team 1 (likely Knicks): {team_counts.get(1, 0)} detections")
print()

print("📈 Movement Analysis:")
print(f"• Average velocity X: {df['vx'].mean():.2f} pixels/frame")
print(f"• Average velocity Y: {df['vy'].mean():.2f} pixels/frame")
speed = (df['vx']**2 + df['vy']**2)**0.5
print(f"• Max speed: {speed.max():.1f} pixels/frame")
print()

print("🎯 Player Performance:")
player_stats = df.groupby('player_id').agg({
    'frame': 'count',
    'confidence': 'mean',
    'x': ['min', 'max'],
    'y': ['min', 'max']
}).round(2)

print("Player tracking summary (top 5 most detected):")
top_players = df['player_id'].value_counts().head()
for pid, count in top_players.items():
    avg_conf = df[df['player_id'] == pid]['confidence'].mean()
    team = df[df['player_id'] == pid]['team'].iloc[0]
    print(f"  Player {pid} (Team {team}): {count} detections, {avg_conf:.2f} avg confidence")

print()
print("📍 Court Coverage:")
print(f"• X-axis range: {df['x'].min():.0f} - {df['x'].max():.0f} pixels")
print(f"• Y-axis range: {df['y'].min():.0f} - {df['y'].max():.0f} pixels")
print(f"• Court utilization: {((df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min())):.0f} pixel area")
