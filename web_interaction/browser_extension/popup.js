// Popup script for browser extension

let isConnected = false;
let currentTabs = [];

document.getElementById('connectBtn').addEventListener('click', connectToOmniAgent);
document.getElementById('refreshBtn').addEventListener('click', refreshTabs);
document.getElementById('screenshotBtn').addEventListener('click', takeScreenshot);
document.getElementById('analyzeBtn').addEventListener('click', analyzeCurrentPage);

// Connect to OmniAgent
function connectToOmniAgent() {
    chrome.runtime.sendMessage({ type: 'connect' }, (response) => {
        if (response && response.success) {
            updateConnectionStatus(true);
            refreshTabs();
        } else {
            alert('Failed to connect to OmniAgent. Make sure the server is running on ws://localhost:8765');
        }
    });
}

// Update connection status UI
function updateConnectionStatus(connected) {
    isConnected = connected;
    const statusDiv = document.getElementById('status');
    const connectionInfo = document.getElementById('connectionInfo');
    const refreshBtn = document.getElementById('refreshBtn');
    const screenshotBtn = document.getElementById('screenshotBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (connected) {
        statusDiv.textContent = 'Connected to OmniAgent';
        statusDiv.className = 'status connected';
        connectionInfo.style.display = 'block';
        refreshBtn.disabled = false;
        screenshotBtn.disabled = false;
        analyzeBtn.disabled = false;
    } else {
        statusDiv.textContent = 'Disconnected from OmniAgent';
        statusDiv.className = 'status disconnected';
        connectionInfo.style.display = 'none';
        refreshBtn.disabled = true;
        screenshotBtn.disabled = true;
        analyzeBtn.disabled = true;
    }
}

// Refresh list of tabs
function refreshTabs() {
    chrome.tabs.query({}, (tabs) => {
        currentTabs = tabs;
        updateTabList(tabs);
        updateTabCount(tabs.length);
    });
}

// Update tab list UI
function updateTabList(tabs) {
    const tabListDiv = document.getElementById('tabList');
    tabListDiv.innerHTML = '';
    tabListDiv.style.display = 'block';
    
    tabs.forEach(tab => {
        const tabItem = document.createElement('div');
        tabItem.className = 'tab-item';
        if (tab.active) {
            tabItem.classList.add('active');
        }
        
        tabItem.innerHTML = `
            <strong>${tab.title || 'Untitled'}</strong><br>
            <small>${tab.url || 'No URL'}</small>
        `;
        
        tabItem.addEventListener('click', () => {
            chrome.tabs.update(tab.id, { active: true });
            refreshTabs();
        });
        
        tabListDiv.appendChild(tabItem);
    });
}

// Update tab count
function updateTabCount(count) {
    document.getElementById('tabCount').textContent = count;
}

// Take screenshot of current tab
function takeScreenshot() {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]) {
            chrome.runtime.sendMessage({
                type: 'take_screenshot',
                tabId: tabs[0].id
            });
            
            alert('Screenshot sent to OmniAgent');
        }
    });
}

// Analyze current page
function analyzeCurrentPage() {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]) {
            chrome.runtime.sendMessage({
                type: 'analyze_page',
                tabId: tabs[0].id
            });
            
            alert('Page analysis sent to OmniAgent');
        }
    });
}

// Check connection status on load
chrome.runtime.sendMessage({ type: 'get_connection_status' }, (response) => {
    if (response && response.connected) {
        updateConnectionStatus(true);
        refreshTabs();
    }
});

// Listen for connection status updates
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'connection_status') {
        updateConnectionStatus(message.connected);
    }
    
    if (message.type === 'tabs_updated') {
        refreshTabs();
    }
});
