(function () {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
        }
      });
    },
    { threshold: 0.15 },
  );

  document.querySelectorAll(".reveal").forEach((element) => {
    observer.observe(element);
  });

  const tabButtons = Array.from(document.querySelectorAll("[data-tab-target]"));
  if (tabButtons.length > 0) {
    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const target = button.getAttribute("data-tab-target");

        tabButtons.forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((panel) => {
          panel.classList.remove("active");
        });

        button.classList.add("active");
        const targetPanel = document.getElementById(target);
        if (targetPanel) {
          targetPanel.classList.add("active");
        }
      });
    });
  }
})();
