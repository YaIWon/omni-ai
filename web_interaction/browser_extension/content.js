// Content script injected into every page

console.log('OmniAgent content script loaded');

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
        case 'get_page_state':
            sendResponse(getPageState());
            break;
            
        case 'execute_script':
            try {
                const result = eval(message.script);
                sendResponse({ success: true, result: result });
            } catch (error) {
                sendResponse({ success: false, error: error.message });
            }
            break;
            
        case 'fill_form':
            fillForm(message.formData);
            sendResponse({ success: true });
            break;
    }
    
    return true; // Keep message channel open for async response
});

// Get comprehensive page state
function getPageState() {
    const state = {
        url: window.location.href,
        title: document.title,
        timestamp: new Date().toISOString(),
        
        // Document info
        doctype: document.doctype ? document.doctype.name : null,
        characterSet: document.characterSet,
        
        // Page dimensions
        viewport: {
            width: window.innerWidth,
            height: window.innerHeight,
            scrollX: window.scrollX,
            scrollY: window.scrollY
        },
        
        // Extract elements
        elements: extractElements(),
        
        // Performance timing
        timing: performance.timing ? {
            loadTime: performance.timing.loadEventEnd - performance.timing.navigationStart,
            domReady: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart
        } : null,
        
        // User agent
        userAgent: navigator.userAgent,
        
        // Cookies (if accessible)
        cookies: document.cookie.split(';').map(c => c.trim()),
        
        // Local storage (if accessible)
        localStorage: Object.keys(localStorage).reduce((obj, key) => {
            obj[key] = localStorage.getItem(key);
            return obj;
        }, {}),
        
        // Form data
        forms: extractForms(),
        
        // Links
        links: extractLinks(),
        
        // Images
        images: extractImages()
    };
    
    return state;
}

// Extract all interactive elements
function extractElements() {
    const elements = [];
    const selectors = [
        'form', 'input', 'textarea', 'select', 'button',
        'a', 'img', 'table', 'iframe', 'video', 'audio'
    ];
    
    selectors.forEach(selector => {
        const found = document.querySelectorAll(selector);
        found.forEach((element, index) => {
            if (index < 50) { // Limit to 50 per type
                elements.push({
                    tag: element.tagName.toLowerCase(),
                    id: element.id || null,
                    name: element.name || null,
                    type: element.type || null,
                    value: element.value || element.textContent?.slice(0, 200) || null,
                    placeholder: element.placeholder || null,
                    className: element.className || null,
                    href: element.href || null,
                    src: element.src || null,
                    alt: element.alt || null,
                    title: element.title || null,
                    selector: generateSelector(element)
                });
            }
        });
    });
    
    return elements;
}

// Generate CSS selector for element
function generateSelector(element) {
    if (element.id) {
        return '#' + element.id;
    }
    
    let selector = element.tagName.toLowerCase();
    
    if (element.className) {
        const classes = element.className.split(' ').filter(c => c);
        if (classes.length > 0) {
            selector += '.' + classes.join('.');
        }
    }
    
    // Add nth-child if needed
    const parent = element.parentElement;
    if (parent) {
        const siblings = Array.from(parent.children);
        const index = siblings.indexOf(element) + 1;
        if (index > 1 || siblings.length > 1) {
            selector += `:nth-child(${index})`;
        }
    }
    
    return selector;
}

// Extract form data
function extractForms() {
    const forms = [];
    document.querySelectorAll('form').forEach((form, index) => {
        if (index < 10) { // Limit to 10 forms
            const formData = {
                id: form.id || null,
                name: form.name || null,
                action: form.action || null,
                method: form.method || 'get',
                elements: []
            };
            
            form.querySelectorAll('input, textarea, select, button').forEach(input => {
                formData.elements.push({
                    tag: input.tagName.toLowerCase(),
                    type: input.type || null,
                    name: input.name || null,
                    value: input.value || null,
                    placeholder: input.placeholder || null,
                    required: input.required || false
                });
            });
            
            forms.push(formData);
        }
    });
    
    return forms;
}

// Extract links
function extractLinks() {
    const links = [];
    document.querySelectorAll('a[href]').forEach((link, index) => {
        if (index < 100) { // Limit to 100 links
            links.push({
                text: link.textContent?.trim() || '',
                href: link.href,
                title: link.title || null
            });
        }
    });
    
    return links;
}

// Extract images
function extractImages() {
    const images = [];
    document.querySelectorAll('img').forEach((img, index) => {
        if (index < 50) { // Limit to 50 images
            images.push({
                src: img.src,
                alt: img.alt || null,
                width: img.width,
                height: img.height,
                naturalWidth: img.naturalWidth,
                naturalHeight: img.naturalHeight
            });
        }
    });
    
    return images;
}

// Fill form with data
function fillForm(formData) {
    for (const [selector, value] of Object.entries(formData)) {
        const element = document.querySelector(selector);
        if (element) {
            element.value = value;
            
            // Trigger events
            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
}

// Monitor DOM changes
const observer = new MutationObserver((mutations) => {
    // Send updates when significant DOM changes occur
    mutations.forEach((mutation) => {
        if (mutation.addedNodes.length > 0 || mutation.removedNodes.length > 0) {
            // Send page state update
            chrome.runtime.sendMessage({
                type: 'dom_changed',
                state: getPageState()
            });
        }
    });
});

observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    characterData: true
});

// Listen for form submissions
document.addEventListener('submit', (event) => {
    const form = event.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);
    
    chrome.runtime.sendMessage({
        type: 'form_submitted',
        formId: form.id || null,
        formName: form.name || null,
        data: data
    });
});

// Send initial page state
chrome.runtime.sendMessage({
    type: 'page_loaded',
    state: getPageState()
});
