async function toggleOption(option) {
  const response = await fetch(`/toggle_option`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ option }),
  });
  const result = await response.json();

  updateStatusMessage(result.message);
}

async function switchVideo(video) {
  const response = await fetch(`/switch_video`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ video }),
  });
  const result = await response.json();

  updateStatusMessage(result.message);

  // Reload the video stream
  const videoElement = document.querySelector('img');
  videoElement.src = `/video_feed?${new Date().getTime()}`; // Add a timestamp to force reload
}

function updateStatusMessage(message) {
  const statusElement = document.getElementById('status-message');
  statusElement.textContent = message;
  statusElement.style.display = 'block';
}

function updateClassCounts() {
  fetch('/class_counts')
    .then(response => response.json())
    .then(data => {
      const container = document.getElementById('class-counts');
      if (Object.keys(data).length === 0) {
        container.textContent = "No detections yet.";
      } else {
        container.innerHTML = Object.entries(data)
          .map(([cls, count]) => `<strong>${cls}</strong>: ${count}`)
          .join(' &nbsp; ');
      }
    });
}
setInterval(updateClassCounts, 1000);
updateClassCounts();
