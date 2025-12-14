// Background service worker for OmniAgent extension

let websocket = null;
let isConnected = false;
let tabStates = new Map();

// Connect to OmniAgent WebSocket server
function connectToOmniAgent() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        return;
    }

    websocket = new WebSocket('ws://localhost:8765');
    
    websocket.onopen = function() {
        console.log('Connected to OmniAgent');
        isConnected = true;
        
        // Send initial state
        sendAllTabsState();
    };
    
    websocket.onmessage = async function(event) {
        try {
            const message = JSON.parse(event.data);
            await handleOmniAgentMessage(message);
        } catch (error) {
            console.error('Error processing message:', error);
        }
    };
    
    websocket.onclose = function() {
        console.log('Disconnected from OmniAgent');
        isConnected = false;
        
        // Try to reconnect after 5 seconds
        setTimeout(connectToOmniAgent, 5000);
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
}

// Send state of all tabs
async function sendAllTabsState() {
    const tabs = await chrome.tabs.query({});
    
    for (const tab of tabs) {
        await sendTabState(tab.id);
    }
}

// Send state of specific tab
async function sendTabState(tabId) {
    try {
        const tab = await chrome.tabs.get(tabId);
        
        // Get page info
        const tabInfo = {
            tabId: tab.id,
            url: tab.url,
            title: tab.title,
            active: tab.active,
            windowId: tab.windowId,
            timestamp: new Date().toISOString()
        };
        
        // Send to OmniAgent
        if (isConnected && websocket) {
            websocket.send(JSON.stringify({
                type: 'tab_state',
                data: tabInfo
            }));
        }
        
        // Store state
        tabStates.set(tabId, tabInfo);
        
    } catch (error) {
        console.error('Error getting tab state:', error);
    }
}

// Handle messages from OmniAgent
async function handleOmniAgentMessage(message) {
    switch (message.type) {
        case 'get_tab_state':
            await sendTabState(message.tabId);
            break;
            
        case 'execute_script':
            await executeScript(message);
            break;
            
        case 'navigate':
            await navigateTab(message);
            break;
            
        case 'fill_form':
            await fillForm(message);
            break;
            
        case 'take_screenshot':
            await takeScreenshot(message);
            break;
            
        case 'extract_data':
            await extractData(message);
            break;
    }
}

// Execute script in tab
async function executeScript(message) {
    try {
        await chrome.scripting.executeScript({
            target: { tabId: message.tabId },
            func: (script) => {
                return eval(script);
            },
            args: [message.script]
        });
        
        // Send success response
        if (websocket) {
            websocket.send(JSON.stringify({
                type: 'script_executed',
                tabId: message.tabId,
                success: true
            }));
        }
    } catch (error) {
        console.error('Error executing script:', error);
    }
}

// Navigate tab to URL
async function navigateTab(message) {
    try {
        await chrome.tabs.update(message.tabId, { url: message.url });
    } catch (error) {
        console.error('Error navigating tab:', error);
    }
}

// Fill form in tab
async function fillForm(message) {
    try {
        await chrome.scripting.executeScript({
            target: { tabId: message.tabId },
            func: (formData) => {
                // Fill each field
                for (const [selector, value] of Object.entries(formData)) {
                    const element = document.querySelector(selector);
                    if (element) {
                        element.value = value;
                        
                        // Trigger change events
                        element.dispatchEvent(new Event('input', { bubbles: true }));
                        element.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            },
            args: [message.formData]
        });
    } catch (error) {
        console.error('Error filling form:', error);
    }
}

// Take screenshot of tab
async function takeScreenshot(message) {
    try {
        const screenshot = await chrome.tabs.captureVisibleTab(
            message.windowId || chrome.windows.WINDOW_ID_CURRENT,
            { format: 'png' }
        );
        
        // Send screenshot to OmniAgent
        if (websocket) {
            websocket.send(JSON.stringify({
                type: 'screenshot',
                tabId: message.tabId,
                dataUrl: screenshot
            }));
        }
    } catch (error) {
        console.error('Error taking screenshot:', error);
    }
}

// Extract data from tab
async function extractData(message) {
    try {
        const results = await chrome.scripting.executeScript({
            target: { tabId: message.tabId },
            func: (pattern) => {
                const data = {};
                
                for (const [key, selector] of Object.entries(pattern)) {
                    const elements = document.querySelectorAll(selector);
                    if (elements.length === 1) {
                        data[key] = elements[0].textContent.trim();
                    } else if (elements.length > 1) {
                        data[key] = Array.from(elements).map(el => el.textContent.trim());
                    }
                }
                
                return data;
            },
            args: [message.pattern]
        });
        
        // Send extracted data
        if (websocket && results[0]) {
            websocket.send(JSON.stringify({
                type: 'extracted_data',
                tabId: message.tabId,
                data: results[0].result
            }));
        }
    } catch (error) {
        console.error('Error extracting data:', error);
    }
}

// Listen for tab updates
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete') {
        await sendTabState(tabId);
    }
});

chrome.tabs.onActivated.addListener(async (activeInfo) => {
    await sendTabState(activeInfo.tabId);
});

chrome.tabs.onRemoved.addListener((tabId) => {
    tabStates.delete(tabId);
    
    // Notify OmniAgent
    if (websocket) {
        websocket.send(JSON.stringify({
            type: 'tab_removed',
            tabId: tabId
        }));
    }
});

// Initialize connection
connectToOmniAgent();
