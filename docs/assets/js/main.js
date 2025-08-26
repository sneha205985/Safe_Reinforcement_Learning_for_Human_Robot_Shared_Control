/**
 * Main JavaScript file for Safe RL Documentation Site
 * Handles interactive functionality and user experience enhancements
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize all functionality
    initScrollProgress();
    initSearchFunctionality();
    initMathJaxConfiguration();
    initCodeCopyButtons();
    initImageModal();
    initTableOfContents();
    initSmoothScrolling();
    initReadingTime();
    
});

/**
 * Initialize scroll progress indicator
 */
function initScrollProgress() {
    const progressBar = document.getElementById('progress-bar');
    if (!progressBar) return;
    
    window.addEventListener('scroll', function() {
        const scrollTotal = document.documentElement.scrollHeight - window.innerHeight;
        const scrollCurrent = window.pageYOffset;
        const scrollPercent = (scrollCurrent / scrollTotal) * 100;
        
        progressBar.style.width = scrollPercent + '%';
    });
}

/**
 * Initialize search functionality
 */
function initSearchFunctionality() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    
    if (!searchInput || !searchResults) return;
    
    let searchData = [];
    let searchTimeout;
    
    // Load search data (you would typically load this from a JSON file)
    loadSearchData().then(data => {
        searchData = data;
    });
    
    // Search input event handler
    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        const query = this.value.trim();
        
        if (query.length < 3) {
            searchResults.style.display = 'none';
            return;
        }
        
        // Debounce search
        searchTimeout = setTimeout(() => {
            performSearch(query, searchData, searchResults);
        }, 300);
    });
    
    // Hide search results when clicking outside
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.style.display = 'none';
        }
    });
    
    // Show search results when focusing on input
    searchInput.addEventListener('focus', function() {
        if (searchResults.children.length > 0) {
            searchResults.style.display = 'block';
        }
    });
}

/**
 * Load search data (mock implementation)
 */
async function loadSearchData() {
    // In a real implementation, this would fetch from a generated JSON file
    return [
        {
            title: "Constrained Policy Optimization",
            url: "/pages/methodology.html#cpo",
            excerpt: "Mathematical formulation and implementation of CPO algorithm..."
        },
        {
            title: "Safety Constraints",
            url: "/pages/methodology.html#safety",
            excerpt: "Types of safety constraints and their mathematical representation..."
        },
        {
            title: "Training Results",
            url: "/pages/results.html#training",
            excerpt: "Performance analysis and learning curve results..."
        },
        {
            title: "Baseline Comparisons",
            url: "/pages/results.html#baselines",
            excerpt: "Statistical comparisons with PPO, TRPO, and other methods..."
        }
    ];
}

/**
 * Perform search and display results
 */
function performSearch(query, searchData, searchResults) {
    const results = searchData.filter(item => 
        item.title.toLowerCase().includes(query.toLowerCase()) ||
        item.excerpt.toLowerCase().includes(query.toLowerCase())
    );
    
    searchResults.innerHTML = '';
    
    if (results.length === 0) {
        const noResults = document.createElement('div');
        noResults.className = 'search-result';
        noResults.innerHTML = '<div class="result-title">No results found</div>';
        searchResults.appendChild(noResults);
    } else {
        results.slice(0, 5).forEach(result => {
            const resultElement = document.createElement('div');
            resultElement.className = 'search-result';
            resultElement.innerHTML = `
                <div class="result-title">${highlightMatch(result.title, query)}</div>
                <div class="result-excerpt">${highlightMatch(result.excerpt, query)}</div>
            `;
            
            resultElement.addEventListener('click', () => {
                window.location.href = result.url;
            });
            
            searchResults.appendChild(resultElement);
        });
    }
    
    searchResults.style.display = 'block';
}

/**
 * Highlight search matches in text
 */
function highlightMatch(text, query) {
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<strong>$1</strong>');
}

/**
 * Configure MathJax for better rendering
 */
function initMathJaxConfiguration() {
    // Additional MathJax configuration if needed
    if (window.MathJax) {
        MathJax.Hub.Config({
            showProcessingMessages: false,
            messageStyle: "none",
            extensions: ["tex2jax.js"],
            jax: ["input/TeX", "output/HTML-CSS"],
            tex2jax: {
                inlineMath: [["$", "$"], ["\\(", "\\)"]],
                displayMath: [["$$", "$$"], ["\\[", "\\]"]],
                skipTags: ["script", "noscript", "style", "textarea", "pre", "code"]
            },
            "HTML-CSS": {
                availableFonts: ["STIX", "TeX"],
                preferredFont: "TeX",
                styles: {
                    ".MathJax": {
                        color: "#000"
                    }
                }
            }
        });
        
        MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
    }
}

/**
 * Add copy buttons to code blocks
 */
function initCodeCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code, .highlight pre');
    
    codeBlocks.forEach(codeBlock => {
        const pre = codeBlock.tagName === 'PRE' ? codeBlock : codeBlock.parentElement;
        
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = 'Copy';
        copyButton.setAttribute('aria-label', 'Copy code to clipboard');
        
        // Position button
        pre.style.position = 'relative';
        copyButton.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            background: #333;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
            opacity: 0.7;
            transition: opacity 0.3s;
        `;
        
        // Add hover effects
        copyButton.addEventListener('mouseenter', () => {
            copyButton.style.opacity = '1';
        });
        
        copyButton.addEventListener('mouseleave', () => {
            copyButton.style.opacity = '0.7';
        });
        
        // Copy functionality
        copyButton.addEventListener('click', async () => {
            const code = codeBlock.textContent || codeBlock.innerText;
            
            try {
                await navigator.clipboard.writeText(code);
                copyButton.innerHTML = 'Copied!';
                copyButton.style.background = '#28a745';
                
                setTimeout(() => {
                    copyButton.innerHTML = 'Copy';
                    copyButton.style.background = '#333';
                }, 2000);
            } catch (err) {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = code;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                
                copyButton.innerHTML = 'Copied!';
                setTimeout(() => {
                    copyButton.innerHTML = 'Copy';
                }, 2000);
            }
        });
        
        pre.appendChild(copyButton);
    });
}

/**
 * Initialize image modal for enlarged viewing
 */
function initImageModal() {
    const images = document.querySelectorAll('figure img, .content img');
    
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.style.cssText = `
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.9);
        cursor: pointer;
    `;
    
    const modalImg = document.createElement('img');
    modalImg.style.cssText = `
        margin: auto;
        display: block;
        width: auto;
        height: auto;
        max-width: 90%;
        max-height: 90%;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    `;
    
    const modalCaption = document.createElement('div');
    modalCaption.style.cssText = `
        margin: auto;
        display: block;
        width: 80%;
        max-width: 700px;
        text-align: center;
        color: #ccc;
        padding: 10px 0;
        position: absolute;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
    `;
    
    modal.appendChild(modalImg);
    modal.appendChild(modalCaption);
    document.body.appendChild(modal);
    
    // Add click handlers to images
    images.forEach(img => {
        if (img.closest('a')) return; // Skip images that are already links
        
        img.style.cursor = 'pointer';
        img.addEventListener('click', () => {
            modal.style.display = 'block';
            modalImg.src = img.src;
            modalImg.alt = img.alt;
            modalCaption.innerHTML = img.alt || '';
        });
    });
    
    // Close modal when clicked
    modal.addEventListener('click', () => {
        modal.style.display = 'none';
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.style.display === 'block') {
            modal.style.display = 'none';
        }
    });
}

/**
 * Generate table of contents for long pages
 */
function initTableOfContents() {
    const tocContainer = document.getElementById('table-of-contents');
    if (!tocContainer) return;
    
    const headings = document.querySelectorAll('h2, h3, h4');
    if (headings.length === 0) return;
    
    const toc = document.createElement('nav');
    toc.className = 'toc';
    
    const tocList = document.createElement('ul');
    let currentLevel = 2;
    let currentList = tocList;
    
    headings.forEach((heading, index) => {
        const level = parseInt(heading.tagName.charAt(1));
        const id = heading.id || `heading-${index}`;
        
        if (!heading.id) {
            heading.id = id;
        }
        
        const listItem = document.createElement('li');
        const link = document.createElement('a');
        link.href = `#${id}`;
        link.textContent = heading.textContent;
        link.className = `toc-level-${level}`;
        
        // Smooth scroll to heading
        link.addEventListener('click', (e) => {
            e.preventDefault();
            heading.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        });
        
        listItem.appendChild(link);
        
        if (level > currentLevel) {
            const nestedList = document.createElement('ul');
            nestedList.appendChild(listItem);
            currentList.lastElementChild.appendChild(nestedList);
            currentList = nestedList;
        } else if (level < currentLevel) {
            // Find appropriate parent list
            let parentList = tocList;
            for (let i = 2; i < level; i++) {
                parentList = parentList.lastElementChild?.querySelector('ul') || parentList;
            }
            parentList.appendChild(listItem);
            currentList = parentList;
        } else {
            currentList.appendChild(listItem);
        }
        
        currentLevel = level;
    });
    
    const tocTitle = document.createElement('h3');
    tocTitle.textContent = 'Table of Contents';
    tocTitle.style.marginTop = '0';
    
    toc.appendChild(tocTitle);
    toc.appendChild(tocList);
    tocContainer.appendChild(toc);
    
    // Add sticky positioning
    toc.style.cssText = `
        position: sticky;
        top: 20px;
        background: #f8f8f8;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    `;
    
    // Style TOC links
    const tocLinks = toc.querySelectorAll('a');
    tocLinks.forEach(link => {
        link.style.cssText = `
            display: block;
            padding: 5px 0;
            color: #333;
            text-decoration: none;
            border-left: 3px solid transparent;
            padding-left: 10px;
            transition: all 0.2s;
        `;
        
        if (link.className === 'toc-level-3') {
            link.style.paddingLeft = '20px';
            link.style.fontSize = '0.9em';
        } else if (link.className === 'toc-level-4') {
            link.style.paddingLeft = '30px';
            link.style.fontSize = '0.85em';
        }
        
        link.addEventListener('mouseenter', () => {
            link.style.borderLeftColor = '#2a2a2a';
            link.style.backgroundColor = 'rgba(42, 42, 42, 0.05)';
        });
        
        link.addEventListener('mouseleave', () => {
            link.style.borderLeftColor = 'transparent';
            link.style.backgroundColor = 'transparent';
        });
    });
}

/**
 * Initialize smooth scrolling for anchor links
 */
function initSmoothScrolling() {
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
                
                // Update URL without jumping
                history.pushState(null, null, targetId);
            }
        });
    });
}

/**
 * Calculate and display estimated reading time
 */
function initReadingTime() {
    const readingTimeContainer = document.getElementById('reading-time');
    if (!readingTimeContainer) return;
    
    const content = document.querySelector('.page-content, .post-content, main');
    if (!content) return;
    
    const text = content.textContent || content.innerText;
    const wordsPerMinute = 200; // Average reading speed
    const wordCount = text.trim().split(/\s+/).length;
    const readingTime = Math.ceil(wordCount / wordsPerMinute);
    
    readingTimeContainer.innerHTML = `
        <span class="reading-time">
            📖 ${readingTime} min read • ${wordCount.toLocaleString()} words
        </span>
    `;
    
    // Style reading time
    readingTimeContainer.style.cssText = `
        font-size: 0.9em;
        color: #666;
        margin-bottom: 20px;
        padding: 10px;
        background: #f8f8f8;
        border-radius: 5px;
        text-align: center;
    `;
}

/**
 * Utility function to debounce function calls
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        
        if (callNow) func.apply(context, args);
    };
}

/**
 * Lazy loading for images (performance optimization)
 */
function initLazyLoading() {
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });
        
        const images = document.querySelectorAll('img[data-src]');
        images.forEach(img => imageObserver.observe(img));
    }
}

// Initialize lazy loading if there are lazy images
document.addEventListener('DOMContentLoaded', function() {
    if (document.querySelectorAll('img[data-src]').length > 0) {
        initLazyLoading();
    }
});

// Analytics and performance tracking (if needed)
function trackPageView() {
    if (typeof gtag !== 'undefined') {
        gtag('config', 'GA_TRACKING_ID', {
            page_title: document.title,
            page_location: window.location.href
        });
    }
}

// Service Worker registration for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}