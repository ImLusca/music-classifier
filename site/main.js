const tabButtons = Array.from(document.querySelectorAll("[data-tab]"));
const panels = Array.from(document.querySelectorAll("[data-panel]"));
const copyButton = document.querySelector("[data-copy-active]");

function setActiveTab(tabName) {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === tabName);
  });
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const activePanel = document.querySelector(".code-block.active code");
    if (!activePanel) return;

    const originalText = copyButton.textContent.trim();
    try {
      await navigator.clipboard.writeText(activePanel.textContent.trim());
      copyButton.textContent = "Copiado";
    } catch {
      copyButton.textContent = "Selecione o comando";
    }

    window.setTimeout(() => {
      copyButton.innerHTML = '<span aria-hidden="true">⧉</span> Copiar';
      copyButton.setAttribute("aria-label", originalText);
    }, 1600);
  });
}

