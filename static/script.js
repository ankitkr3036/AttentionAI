
const $ = (id) => document.getElementById(id);

let chart = null;
let audioContext = null;

let lastAlertAt = 0;
let alertDismissed = false;
let sessionStarted = false;

const points = [];

// TIME FORMAT
function formatTime(seconds) {
    const total = Math.max(0, Math.floor(seconds || 0));

    return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}


// ATTENTION SCORE COLOR
function getScoreColor(score) {

    if (score >= 80) return "#55d6a2";

    if (score >= 50) return "#ffbd69";

    return "#ff718d";
}


// UPDATE SCORE RING
function updateScoreRing(score) {

    const ring = $("scoreRing");

    if (!ring) return;

    const value = Math.max(0, Math.min(100, Number(score) || 0));

    const degrees = value * 3.6;

    const color = getScoreColor(value);

    ring.style.background = `
        conic-gradient(
            ${color} 0deg,
            #9b83ff ${degrees * 0.65}deg,
            #373454 ${degrees}deg,
            #373454 360deg
        )
    `;

    ring.style.boxShadow = `0 0 30px ${color}33`;
}


// STATUS COLOR
function updateStatusColor(status) {

    const pill = $("statusPill");

    if (!pill) return;

    const colors = {
        "Focused": {
            color: "#55d6a2",
            background: "#153c35"
        },

        "Distracted": {
            color: "#ffbd69",
            background: "#423322"
        },

        "Drowsy": {
            color: "#ff718d",
            background: "#492b36"
        },

        "No face detected": {
            color: "#999bbd",
            background: "#272842"
        }
    };

    const style = colors[status] || colors["No face detected"];

    pill.style.color = style.color;
    pill.style.background = style.background;
}


// INITIALIZE CHART
function initChart() {

    const canvas = $("attentionChart");

    if (!window.Chart || !canvas) return;

    chart = new Chart(canvas, {

        type: "line",

        data: {
            labels: [],

            datasets: [{
                label: "Attention Score",

                data: [],

                borderColor: "#a18aff",

                backgroundColor: "rgba(161,138,255,.12)",

                fill: true,

                tension: 0.4,

                pointRadius: 2,

                pointHoverRadius: 5,

                borderWidth: 2
            }]
        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            animation: {
                duration: 300
            },

            plugins: {
                legend: {
                    display: false
                },

                tooltip: {
                    backgroundColor: "#17182f",

                    titleColor: "#ffffff",

                    bodyColor: "#b8aaff",

                    borderColor: "#45406b",

                    borderWidth: 1
                }
            },

            scales: {

                x: {
                    ticks: {
                        color: "#8586a8",
                        maxTicksLimit: 6
                    },

                    grid: {
                        display: false
                    }
                },

                y: {
                    min: 0,
                    max: 100,

                    ticks: {
                        color: "#8586a8",
                        stepSize: 25
                    },

                    grid: {
                        color: "rgba(255,255,255,.06)"
                    }
                }
            }
        }
    });
}


// AUDIO ALERT
function playAlert() {

    if (
        $("muteToggle")?.checked ||
        Date.now() - lastAlertAt < 8000
    ) {
        return;
    }

    try {

        audioContext ||= new (
            window.AudioContext ||
            window.webkitAudioContext
        )();

        if (audioContext.state === "suspended") {
            audioContext.resume();
        }

        const oscillator = audioContext.createOscillator();

        const gain = audioContext.createGain();

        oscillator.frequency.value = 660;

        gain.gain.setValueAtTime(
            0.0001,
            audioContext.currentTime
        );

        gain.gain.exponentialRampToValueAtTime(
            0.12,
            audioContext.currentTime + 0.03
        );

        gain.gain.exponentialRampToValueAtTime(
            0.0001,
            audioContext.currentTime + 0.45
        );

        oscillator.connect(gain);

        gain.connect(audioContext.destination);

        oscillator.start();

        oscillator.stop(audioContext.currentTime + 0.46);

        lastAlertAt = Date.now();

    } catch (error) {

        console.info("Audio alert unavailable:", error);
    }
}


// UPDATE DASHBOARD
function updateStatus(data) {

    const status = data.status || "No face detected";

    const faceDetected = !!data.face_detected;

    const score = faceDetected
        ? Math.max(0, Math.min(100, Number(data.attention_score) || 0))
        : null;

    // STATUS
    $("statusPill").textContent = status.toUpperCase();

    updateStatusColor(status);

    // ATTENTION SCORE
    $("scoreValue").textContent =
        score === null ? "--" : `${score}%`;

    updateScoreRing(score ?? 0);

    // METRICS
    $("blinkValue").textContent = data.blink_count ?? 0;

    $("earValue").textContent =
        data.ear == null
            ? "--"
            : Number(data.ear).toFixed(3);

    $("closedValue").textContent =
        `${data.eyes_closed_seconds || 0}s`;

    $("gazeValue").textContent =
        data.gaze || "Unknown";

    $("headValue").textContent =
        data.head_pose || "Unknown";

    $("durationValue").textContent =
        formatTime(data.session_duration);

    $("streakValue").textContent =
        formatTime(data.focus_streak);


    // SCORE DESCRIPTION
    if (data.message) {

        $("scoreCaption").textContent = data.message;

    } else if (status === "Focused") {

        $("scoreCaption").textContent =
            "Looking steady. Keep your rhythm.";

    } else if (status === "Distracted") {

        $("scoreCaption").textContent =
            "Your gaze or head direction is away from center.";

    } else if (status === "Drowsy") {

        $("scoreCaption").textContent =
            "Prolonged eye closure detected.";

    } else {

        $("scoreCaption").textContent =
            "Waiting for a face.";
    }


    // CAMERA STATUS
    $("cameraLabel").textContent =
        data.camera_available
            ? "Camera connected"
            : (data.session_active
                ? "Camera unavailable"
                : "Camera idle");

    $("cameraDot").parentElement.classList.toggle(
        "connected",
        !!data.camera_available
    );


    // START AND STOP BUTTONS
    $("startBtn").disabled = !!data.session_active;

    $("stopBtn").disabled = !data.session_active;


    // CAMERA VIDEO
    if (data.session_active && !sessionStarted) {

        $("videoFeed").src =
            `/video_feed?t=${Date.now()}`;

        $("videoFeed").style.display = "block";

        $("videoPlaceholder").classList.add("hidden");

        sessionStarted = true;
    }

    if (!data.session_active && sessionStarted) {

        $("videoFeed").removeAttribute("src");

        $("videoFeed").style.display = "none";

        $("videoPlaceholder").classList.remove("hidden");

        sessionStarted = false;
    }


    // ALERT BANNER
    const banner = $("alertBanner");

    if (data.alert && !alertDismissed) {

        banner.classList.remove("hidden");

        playAlert();

    } else if (!data.alert) {

        banner.classList.add("hidden");

        alertDismissed = false;
    }


    // REAL-TIME CHART
    if (chart && faceDetected && score !== null) {

        points.push({

            label: new Date().toLocaleTimeString(
                [],
                {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit"
                }
            ),

            score: score
        });

        if (points.length > 40) {
            points.shift();
        }

        chart.data.labels = points.map(p => p.label);

        chart.data.datasets[0].data =
            points.map(p => p.score);

        chart.update("none");
    }
}


// REFRESH API
async function refresh() {

    try {

        const response = await fetch("/api/status");

        if (!response.ok) {
            throw new Error("Server response error");
        }

        const data = await response.json();

        updateStatus(data);

    } catch (error) {

        $("cameraLabel").textContent = "Server disconnected";

        $("cameraDot").parentElement.classList.remove("connected");

        console.warn("Status refresh failed:", error);
    }
}


// LOAD SESSION HISTORY
async function loadHistory() {

    try {

        const response = await fetch("/api/history");

        if (!response.ok) {
            throw new Error("History loading failed");
        }

        const data = await response.json();

        const container = $("historyList");

        if (!container) return;

        if (!data.sessions || !data.sessions.length) {

            container.innerHTML =
                '<p class="empty">No sessions saved yet.</p>';

            return;
        }

        container.innerHTML = data.sessions.map(s => `

            <div class="history-item">

                <div>
                    <strong>Session #${s.id}</strong>
                    <br>

                    <span>
                        ${new Date(s.started_at + "Z").toLocaleString()}
                    </span>
                </div>

                <div style="text-align:right">

                    <strong>${s.average_score}/100</strong>

                    <br>

                    <span>
                        ${formatTime(s.duration_seconds)}
                    </span>

                </div>

            </div>

        `).join("");

    } catch (error) {

        console.warn(
            "Could not load session history:",
            error
        );
    }
}


// POST REQUEST
async function post(url) {

    const response = await fetch(url, {
        method: "POST"
    });

    const data = await response.json();

    if (!response.ok || data.ok === false) {

        throw new Error(
            data.error || "Request failed"
        );
    }

    return data;
}


// START SESSION
$("startBtn").addEventListener("click", async () => {

    try {

        await post("/api/start");

        points.length = 0;

        alertDismissed = false;

        if (chart) {

            chart.data.labels = [];

            chart.data.datasets[0].data = [];

            chart.update();
        }

        await refresh();

    } catch (error) {

        alert(error.message);
    }
});


// STOP SESSION
$("stopBtn").addEventListener("click", async () => {

    try {

        await post("/api/stop");

        await refresh();

        await loadHistory();

    } catch (error) {

        alert(error.message);
    }
});


// RESET SESSION
$("resetBtn").addEventListener("click", async () => {

    if (!confirm("Reset the current session view?")) {
        return;
    }

    try {

        await post("/api/reset");

        points.length = 0;

        alertDismissed = false;

        if (chart) {

            chart.data.labels = [];

            chart.data.datasets[0].data = [];

            chart.update();
        }

        await refresh();

    } catch (error) {

        alert(error.message);
    }
});


// DISMISS ALERT
$("dismissAlert").addEventListener("click", () => {

    alertDismissed = true;

    $("alertBanner").classList.add("hidden");
});


// CLEAR HISTORY
$("clearHistoryBtn").addEventListener("click", async () => {

    if (
        !confirm(
            "Permanently clear all locally stored session history?"
        )
    ) {
        return;
    }

    try {

        await post("/api/history/clear");

        await loadHistory();

    } catch (error) {

        alert(error.message);
    }
});


// INITIALIZE DASHBOARD
initChart();

refresh();

loadHistory();


// AUTO REFRESH
setInterval(refresh, 1000);

setInterval(loadHistory, 15000);
