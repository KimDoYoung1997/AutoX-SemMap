import numpy as np

# ✅ 사전에 지정한 두 개의 경로점
custom_traj = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# ✅ 'custom_traj.npz' 파일로 저장
np.savez("custom_traj.npz", traj=custom_traj)

print("✅ 'custom_traj.npz' 파일이 성공적으로 생성되었습니다.")

with np.load("custom_traj.npz") as data:
    loaded_traj = data["traj"]

print("📍 저장된 경로점:\n", loaded_traj)
