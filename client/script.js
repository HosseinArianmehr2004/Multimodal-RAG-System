// --- DOM Elements ---
const chatMessages = document.getElementById("chatMessages");
const contextContent = document.getElementById("contextContent");
const messageInput = document.getElementById("messageInput");
const fileInput = document.getElementById("fileInput");
const filePreview = document.getElementById("filePreview");
const modeIndicator = document.getElementById("modeIndicator");
const attachBtn = document.getElementById("attachBtn");
const sendBtn = document.getElementById("sendBtn");


let attachedFiles = [];


function removeWelcomeMessage() {
    const welcome = chatMessages.querySelector(".welcome-message");
    if (welcome) welcome.remove();
}

function addBotMessage(text) {
    removeWelcomeMessage();

    const el = document.createElement("div");
    el.className = "message bot";

    const content = document.createElement("div");
    content.className = "message-content";

    // --- RENDER MARKDOWN ---
    content.innerHTML = marked.parse(text);

    el.appendChild(content);
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return el;
}

function displayUserMessage(text, files) {
    removeWelcomeMessage();

    const el = document.createElement("div");
    el.className = "message user";

    const content = document.createElement("div");
    content.className = "message-content";

    // 1. Add Text if it exists
    if (text) {
        const textEl = document.createElement("div");
        textEl.className = "message-text"; // Use the CSS class from styles.css
        textEl.textContent = text;
        content.appendChild(textEl);
    }

    // 2. Add Files if they exist
    if (files && files.length > 0) {
        const filesEl = document.createElement("div");
        filesEl.className = "message-files"; // Use the CSS class

        files.forEach(file => {
            const fileItemEl = document.createElement("div");
            fileItemEl.className = "message-file-item"; // Use the CSS class

            let previewEl;
            if (file.type.startsWith("image/")) {
                previewEl = document.createElement("img");
                previewEl.className = "message-file-preview";
                previewEl.src = URL.createObjectURL(file);
            } else if (file.type.startsWith("audio/")) {
                previewEl = document.createElement("audio");
                previewEl.controls = true;
                previewEl.src = URL.createObjectURL(file);
            } else {
                // Fallback for other file types
                previewEl = document.createElement("span");
                previewEl.className = "file-type-icon";
                previewEl.textContent = "📄"; // Generic doc icon
            }

            const fileInfo = document.createElement("div");
            fileInfo.className = "file-info";
            const fileName = document.createElement("div");
            fileName.className = "file-name"; // Use the CSS class
            fileName.textContent = file.name;
            fileInfo.appendChild(fileName);

            fileItemEl.appendChild(previewEl);
            fileItemEl.appendChild(fileInfo);
            filesEl.appendChild(fileItemEl);
        });
        content.appendChild(filesEl);
    }

    el.appendChild(content);
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateFilePreview() {
    filePreview.innerHTML = ""; // Clear existing previews

    if (attachedFiles.length === 0) {
        filePreview.classList.remove("active"); // Hide the preview area
        modeIndicator.textContent = "💬 Text Mode";
        return;
    }

    filePreview.classList.add("active"); // Show the preview area
    modeIndicator.textContent = `📎 ${attachedFiles.length} file(s) attached`;

    attachedFiles.forEach((file, index) => {
        const item = document.createElement("div");
        item.className = "preview-item"; // Use CSS class

        let preview;
        if (file.type.startsWith("image/")) {
            preview = document.createElement("img");
            preview.src = URL.createObjectURL(file);
        } else if (file.type.startsWith("audio/")) {
            preview = document.createElement("audio");
            preview.controls = true;
            preview.src = URL.createObjectURL(file);
        } else {
            // Fallback for generic file
            preview = document.createElement("span");
            preview.textContent = "📄";
            preview.style.fontSize = "40px"; // Make icon visible
        }

        const name = document.createElement("span");
        name.className = "file-name";
        name.textContent = file.name;

        const removeBtn = document.createElement("button");
        removeBtn.className = "remove-file-btn";
        removeBtn.textContent = "✕"; // 'x' icon
        removeBtn.title = "Remove file";

        removeBtn.onclick = () => {
            attachedFiles.splice(index, 1); // Remove file from array
            fileInput.value = ""; // Reset input to allow re-adding
            updateFilePreview(); // Re-render previews
        };

        item.appendChild(preview);
        item.appendChild(name);
        item.appendChild(removeBtn);
        filePreview.appendChild(item);
    });
}

function clearContext() {
    contextContent.innerHTML = `<div class="empty-state"><p>🔍 Retrieved documents will appear here</p></div>`;
}

function displayRetrievedContent(allItems) {
    contextContent.innerHTML = "";

    if (!allItems || allItems.length === 0) {
        clearContext();
        return;
    }

    // Ensure order: text -> image -> audio
    const texts = allItems.filter(i => i.modality === "text");
    const images = allItems.filter(i => i.modality === "image");
    const audios = allItems.filter(i => i.modality === "audio");

    // TEXTS
    texts.forEach(item => {
        const box = document.createElement("div");
        box.className = "content-box text-box";
        box.innerHTML = `
            <div>📝 <strong>${escapeHtml(item.content || "")}</strong></div>
            <div class="content-id">${escapeHtml(item.contentId || "")}</div>
        `;
        contextContent.appendChild(box);
    });

    // IMAGES
    images.forEach(item => {
        const box = document.createElement("div");
        box.className = "content-box image-box";
        const filePath = item.filePath || "";
        box.innerHTML = `
            <img src="${escapeAttr(filePath)}" alt="retrieved image" />
            <div class="image-meta">
                <div>🖼 Image</div>
                <div class="content-id">${escapeHtml(item.contentId || "")}</div>
            </div>
        `;
        contextContent.appendChild(box);
    });

    // AUDIOS
    audios.forEach(item => {
        const box = document.createElement("div");
        box.className = "content-box audio-box";
        const filePath = item.filePath || "";
        box.innerHTML = `
            <audio controls preload="none">
                <source src="${escapeAttr(filePath)}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
            <div class="audio-meta">
                <div>🎵 Audio</div>
                <div class="content-id">${escapeHtml(item.contentId || "")}</div>
            </div>
        `;
        contextContent.appendChild(box);
    });

    // scroll to top of the left panel
    contextContent.scrollTop = 0;
}

function escapeHtml(s) {
    if (!s) return "";
    return s.replace(/[&<>"'`=\/]/g, function (c) {
        return ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
            '/': '&#x2F;',
            '`': '&#x60;',
            '=': '&#x3D;'
        })[c];
    });
}

function escapeAttr(s) {
    return escapeHtml(s);
}

function displayLLMReferenceData(llmData) {
    if (!llmData) return;

    const refEl = document.createElement("div");
    refEl.className = "llm-reference";
    refEl.innerHTML = `<strong>📌 Reference Data Sent to LLM:</strong><br>`;

    // Container for horizontal layout
    const container = document.createElement("div");
    container.className = "llm-reference-container";

    // RETRIEVED TEXTS
    if (llmData.retrieved_texts) {
        llmData.retrieved_texts.forEach((txt, idx) => {
            const box = document.createElement("div");
            box.className = "llm-item text-item";
            box.innerHTML = `📄 Text ${idx + 1}: \n${escapeHtml(txt)}`;
            container.appendChild(box);
        });
    }

    // RETRIEVED IMAGES
    if (llmData.retrieved_images) {
        llmData.retrieved_images.forEach((imgPath, idx) => {
            if (imgPath) {
                const box = document.createElement("div");
                box.className = "llm-item image-item";

                const img = document.createElement("img");
                img.src = imgPath;
                img.alt = `Retrieved Image ${idx + 1}`;
                img.className = "llm-image";

                box.appendChild(img);
                container.appendChild(box);
            }
        });
    }

    // RETRIEVED AUDIOS
    if (llmData.retrieved_audios) {
        llmData.retrieved_audios.forEach((audioPath, idx) => {
            if (audioPath) {
                const box = document.createElement("div");
                box.className = "llm-item audio-item";

                const audio = document.createElement("audio");
                audio.controls = true;
                audio.preload = "none";
                audio.className = "llm-audio";

                const source = document.createElement("source");
                source.src = audioPath;
                source.type = "audio/mpeg";
                audio.appendChild(source);

                box.appendChild(audio);
                container.appendChild(box);
            }
        });
    }

    refEl.appendChild(container);
    chatMessages.appendChild(refEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

attachBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        // Add all newly selected files to our array
        attachedFiles.push(...fileInput.files);
    }
    updateFilePreview(); // Re-render the preview list
});

sendBtn.addEventListener("click", async () => {
    const msg = messageInput.value.trim();

    // Check if there is anything to send
    if (msg.length === 0 && attachedFiles.length === 0) {
        return; // Nothing to send
    }

    // 1. Display the user's message in the chat
    // Pass a *copy* of the array, as we'll clear the original
    displayUserMessage(msg, [...attachedFiles]);

    // 2. Prepare data for sending
    const form = new FormData();
    form.append("query", msg);
    attachedFiles.forEach((file) => {
        // Append each file. The server will receive 'files' as an array.
        form.append("files", file, file.name);
    });

    // 3. Send data
    // We use a single endpoint to handle text and files together.
    // **NOTE:** You will need to update your server to handle this
    // single '/search/multimodal' endpoint.

    const thinkingMsg = addBotMessage("🤖 Thinking..."); // Show loading
    sendBtn.disabled = true;
    // 4. Reset inputs
    messageInput.value = "";
    fileInput.value = "";    // Clear the file input
    attachedFiles = [];      // Clear the state array
    updateFilePreview();     // Clear the preview UI

    try {
        const res = await fetch("/search/multimodal", {
            method: "POST",
            body: form
        });

        if (!res.ok) {
            throw new Error(`Server error: ${res.statusText}`);
        }

        const data = await res.json();

        thinkingMsg.remove(); // Remove "Thinking..."

        // --- Handle Response (from your original logic) ---
        if (data.transcription) {
            addBotMessage(`Transcription: ${data.transcription}`);
        }

        const combined = [
            ...(data.text_results || []),
            ...(data.image_results || []),
            ...(data.audio_results || []),
        ];

        displayRetrievedContent(combined);

        // If your server sends back a direct text answer, display it
        if (data.llm_response) {
            addBotMessage(data.llm_response);
        }
        if (data.llm_data) {
            displayLLMReferenceData(data.llm_data);
        }
        // If no content/answer, give a fallback
        if (!combined.length && !data.llm_response && !data.transcription) {
            addBotMessage("I processed your request, but didn't find any specific content to show.");
        }
        // --- End Response Handling ---

    } catch (err) {
        thinkingMsg.remove(); // Remove "Thinking..."
        addBotMessage('An error occurred: ' + String(err));
        console.error(err);
    }

    // Re-enable send button
    sendBtn.disabled = false;
});

messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault(); // Prevent new line
        sendBtn.click();    // Trigger send
    }
});
