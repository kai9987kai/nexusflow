# Portfolio Review for `kai9987kai`

Generated from an automated shallow audit of every public repository (169 total).

## Cross-Repo Findings

- Total repos reviewed: **169**
- Median repo file count: **5** (many small prototypes)
- Average repo file count: **397.1** (a few large projects raise the mean)
- Primary language mix (top 10):
  - HTML: 79
  - Python: 60
  - Unknown: 5
  - JavaScript: 4
  - C#: 4
  - TypeScript: 3
  - PowerShell: 1
  - Java: 1
  - Visual Basic: 1
  - GLSL: 1
- Portfolio category mix:
  - Web Apps / Generators: 55
  - Python Utilities / Scripts: 25
  - AI Simulations / Games: 21
  - AI / ML Tools: 21
  - Desktop Utilities / GUI: 13
  - Misc / Experiments: 12
  - Games / Simulations: 11
  - Game / Unity: 8
  - DSL / Compilers: 3
- Repeated strengths:
  - Strong prototype throughput across HTML, Python, and simulation/game concepts
  - Frequent experimentation with AI/agent/evolution ideas
  - Increasing sophistication in larger projects (docs, security files, packaging)
- Repeated gaps:
  - Testing and CI are uncommon across most repos
  - Many repos are single-purpose prototypes without standardized packaging
  - Similar patterns repeat (UI wiring, config, simulation loops, export paths)
- Language opportunity:
  - A single DSL that targets **rapid web + simulation + AI experiment prototyping** would reduce boilerplate and make projects easier to evolve into maintainable tools.

## Per-Repository Review (All Repos)

### 1. `1313-Game`

- URL: https://github.com/kai9987kai/1313-Game
- Category: Games / Simulations
- Primary language: Python
- Size/File count: 429 KB / 50 files
- Features: python, game, ci
- Focus: This is a game inspired by the cancelled game 1313 and star wars comics project to make the project a reality its not very good but oh well!! TO DO LIST: ADD FLOORING MAKE NEW CHARACTER DESIGNS ADD LEVELS
- Strengths: README/doc present; CI workflow configured
- Risks: no obvious tests
- Opportunities: headless simulation test harness for gameplay logic; standardize packaging with pyproject.toml

### 2. `3d-animal-simulator-Hybrid-Agent`

- URL: https://github.com/kai9987kai/3d-animal-simulator-Hybrid-Agent
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 90 KB / 8 files
- Features: webapp, html, ai_ml, game
- Focus: DIAMOND Hybrid Agent Simulation Welcome to the DIAMOND Hybrid Agent Simulation repository! This project simulates an advanced environment where agents utilize cutting-edge AI methodologies, including quantum-inspired ...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 3. `3D-Shape-generator`

- URL: https://github.com/kai9987kai/3D-Shape-generator
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 13 KB / 5 files
- Features: python
- Focus: A tiny **Blender (Python / `bpy`) script** that procedurally generates a batch of random 3D primitives (cubes or UV spheres), randomizes their transform + color, and triggers a render. :contentReference[oaicite:0]{ind...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 4. `3d-texture-map-converter`

- URL: https://github.com/kai9987kai/3d-texture-map-converter
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 41 KB / 6 files
- Features: python, desktop_gui
- Focus: A desktop (offline) **PBR texture-map generator** that converts a source image (typically **albedo/base-color**) into common **material/utility maps** with a modern **dark UI**, **side-by-side preview**, **strength co...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 5. `3dTextureMapTester-online`

- URL: https://github.com/kai9987kai/3dTextureMapTester-online
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 2 KB / 1 files
- Features: webapp, html
- Focus: An interactive web application built with Three.js to apply and test various texture maps (Texture, Bump, Opacity, Normal, Displacement, Specular) on 3D objects. Supports custom OBJ uploads and dynamic lighting adjust...
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 6. `Advanced-Navigation-Generator`

- URL: https://github.com/kai9987kai/Advanced-Navigation-Generator
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 4 KB / 2 files
- Features: webapp, html
- Focus: Generate a custom navigation bar for your web page with ease! This tool allows you to create a personalized navigation menu by specifying the navigation type, style, and adding multiple pages with links. - Choose betw...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 7. `Advanced-Page-Builder`

- URL: https://github.com/kai9987kai/Advanced-Page-Builder
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 75 KB / 4 files
- Features: webapp, html
- Focus: The Advanced Page Builder is a web-based tool that allows users to dynamically create a webpage layout using drag-and-drop functionality. It's designed to be intuitive and user-friendly, enabling users to add various ...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 8. `Agent-Mouse-Simulation`

- URL: https://github.com/kai9987kai/Agent-Mouse-Simulation
- Category: AI Simulations / Games
- Primary language: Python
- Size/File count: 58 KB / 8 files
- Features: python, ai_ml, game, desktop_gui
- Focus: Agent Mouse Simulation This project simulates an AI-controlled mouse agent that interacts with the screen using pyautogui. The agent uses a convolutional neural network (CNN) to make decisions based on screen captures...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 9. `Alcohol-Unit-Tracker`

- URL: https://github.com/kai9987kai/Alcohol-Unit-Tracker
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 18 KB / 2 files
- Features: webapp, html
- Focus: The Alcohol Unit Tracker is a simple, interactive web application designed to help users monitor their alcohol consumption. Utilizing a dynamic bar graph, the tracker provides a visual representation of alcohol units ...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 10. `androidClone`

- URL: https://github.com/kai9987kai/androidClone
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 32 KB / 5 files
- Features: webapp, html
- Focus: A tiny “android UI in the browser” prototype. This repo currently ships as a single-page `index.html` that lays out common Android-style UI copy (lock screen time/date, “Swipe up to unlock”, quick settings toggles, an...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 11. `App-archive`

- URL: https://github.com/kai9987kai/App-archive
- Category: Misc / Experiments
- Primary language: Unknown
- Size/File count: 75485 KB / 4 files
- Features: None
- Focus: Archive for my old apps or there souce code
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 12. `Arm-Nerve-Simulation`

- URL: https://github.com/kai9987kai/Arm-Nerve-Simulation
- Category: Games / Simulations
- Primary language: HTML
- Size/File count: 22 KB / 3 files
- Features: webapp, html, game
- Focus: realistic simulation of the arm's nervous system with interactive features Features Anatomy 6 major arm nerves with anatomically accurate paths: Musculocutaneous, Median, Ulnar, Radial, Axillary, and Medial Cutaneous ...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic; deploy preset + environment template

### 13. `Atom-Sim`

- URL: https://github.com/kai9987kai/Atom-Sim
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 1 KB / 1 files
- Features: python
- Focus: This is a education atom simulator currenlty having some elemants but soon more
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 14. `ATS-DOCX`

- URL: https://github.com/kai9987kai/ATS-DOCX
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 25 KB / 3 files
- Features: webapp, html
- Focus: Modern protection against exploits on Applicant Tracking Software Systems
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 15. `AutoClicker`

- URL: https://github.com/kai9987kai/AutoClicker
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 163370 KB / 190 files
- Features: python, desktop_gui
- Focus: It spams clicks on the selected Coordinates, We also have a feature that allows for a list of coordinates to be performed and a mega auto clicker that is faster than the base version but is only limited by your comput...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 16. `AutoUnwrap`

- URL: https://github.com/kai9987kai/AutoUnwrap
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 22 KB / 3 files
- Features: webapp, html
- Focus: Research based new unwrapping algorithm
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 17. `av-testing`

- URL: https://github.com/kai9987kai/av-testing
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 9635 KB / 28 files
- Features: webapp, html, python
- Focus: A **safe antivirus testing harness** for validating endpoint/browser protections using standard, non-malicious test artifacts and checks. This project provides a small Python CLI that can: - Generate the standard **EI...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template; standardize packaging with pyproject.toml

### 18. `Bio-Explorer`

- URL: https://github.com/kai9987kai/Bio-Explorer
- Category: AI / ML Tools
- Primary language: HTML
- Size/File count: 53 KB / 2 files
- Features: webapp, html, ai_ml, desktop_gui
- Focus: **Bio-Explorer** is a **standalone, biology-first protein explorer** that chains major public bioinformatics resources into one workflow: **UniProt → STRING → Reactome → AlphaFold** …so you can go from *“this protein”...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 19. `BIOLAB`

- URL: https://github.com/kai9987kai/BIOLAB
- Category: AI / ML Tools
- Primary language: JavaScript
- Size/File count: 29 KB / 10 files
- Features: webapp, html, ai_ml, desktop_gui, javascript
- Focus: **BIOLAB** is a **browser-based biology sandbox**: a small collection of “BIO innovation tools” implemented as a static web app (HTML/CSS/JS) with multiple mini-labs (cellular automata, genetic algorithms, CRISPR conc...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 20. `Calculators`

- URL: https://github.com/kai9987kai/Calculators
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 22 KB / 5 files
- Features: python
- Focus: A Range Of Different Types Of Calculators
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 21. `ChatBot-Trainer`

- URL: https://github.com/kai9987kai/ChatBot-Trainer
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 2714 KB / 8 files
- Features: webapp, html
- Focus: Chatbot trainer is a web based ai training architecture that uses json data
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 22. `ChatGpt-Powershell`

- URL: https://github.com/kai9987kai/ChatGpt-Powershell
- Category: Misc / Experiments
- Primary language: PowerShell
- Size/File count: 2 KB / 2 files
- Features: None
- Focus: chatgpt in powershell Run in powershell ise as administrator you will also have to change my open ai api key
- Strengths: README/doc present
- Risks: no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build

### 23. `CIFAR-100`

- URL: https://github.com/kai9987kai/CIFAR-100
- Category: DSL / Compilers
- Primary language: Python
- Size/File count: 633181 KB / 48091 files
- Features: webapp, html, python, ai_ml, desktop_gui, javascript, dsl, tests
- Focus: - 🌐 **Runs entirely in the browser** - No server required! - 📦 **ONNX.js** - Uses ONNX Runtime for JavaScript - 🎨 **Modern UI** - Beautiful, responsive interface - 🚀 **100 Classes** - Trained using hybrid NPU+GPU+CPU ...
- Strengths: README/doc present; tests present; experiments with DSL/abstraction
- Risks: No major automated flags
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 24. `codexdecipher`

- URL: https://github.com/kai9987kai/codexdecipher
- Category: Web Apps / Generators
- Primary language: TypeScript
- Size/File count: 13 KB / 13 files
- Features: webapp, html, typescript
- Focus: <div align="center"> <img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" /> </div> This contains everything you need to run your app loca...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 25. `Colour-Cube`

- URL: https://github.com/kai9987kai/Colour-Cube
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 4 KB / 2 files
- Features: webapp, html
- Focus: Imagine instead of a colour wheel you had a 3d cube that you can rotate and click the colour you want. This project showcases a 3D color cube visualization using Three.js. Users can explore different color combination...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 26. `Combined-Agents`

- URL: https://github.com/kai9987kai/Combined-Agents
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 117 KB / 14 files
- Features: webapp, html, ai_ml, game
- Focus: **Overview:** This repository contains a collection of advanced simulation environments focused on multi-agent interactions, adaptive learning, and ecosystem dynamics. Each file represents a unique approach to agent b...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 27. `Cursor-Logo-Tester`

- URL: https://github.com/kai9987kai/Cursor-Logo-Tester
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 19 KB / 4 files
- Features: python
- Focus: Test all the default cursor logos
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 28. `Custom-Scrollbar-Styler---Interactive-CSS-Generator`

- URL: https://github.com/kai9987kai/Custom-Scrollbar-Styler---Interactive-CSS-Generator
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 32 KB / 2 files
- Features: webapp, html
- Focus: Welcome to the Custom Scrollbar Styler repository! This interactive web tool allows users to easily customize the appearance of a scrollbar, including its width, height, border-radius, background color, and box-shadow...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 29. `DIAMOND-Simulation`

- URL: https://github.com/kai9987kai/DIAMOND-Simulation
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 130 KB / 6 files
- Features: webapp, html, ai_ml, game
- Focus: This repository hosts an advanced implementation of the DIAMOND Simulation—a dynamic, interactive model that demonstrates how agents navigate, learn, and adapt within a complex and ever-changing environment. Built wit...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 30. `DiceRoller`

- URL: https://github.com/kai9987kai/DiceRoller
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 23 KB / 12 files
- Features: python
- Focus: Small Dice roller program
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 31. `DNA-Helix`

- URL: https://github.com/kai9987kai/DNA-Helix
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 23 KB / 5 files
- Features: webapp, html
- Focus: A **single-file, standalone HTML/CSS/JS** project that renders a **DNA double-helix** visualization/animation in the browser. - **Repo type:** minimalist / no-build / open-in-browser - **Primary file:** `index.html` -...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 32. `Drag-and-Drop-Animation-Builder`

- URL: https://github.com/kai9987kai/Drag-and-Drop-Animation-Builder
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 24 KB / 2 files
- Features: webapp, html
- Focus: This is a simple web-based animation builder that allows users to create animations through drag-and-drop interactions. The builder includes features such as adding keyframes, deleting keyframes, and playing animation...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 33. `Duckhunt`

- URL: https://github.com/kai9987kai/Duckhunt
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 2896 KB / 12 files
- Features: python, desktop_gui
- Focus: <h1>DuckHunter</h1> <h3>Prevent RubberDucky (or other keystroke injection) attacks</h3> <h3>Try out the new setup GUI, which helps you to set up the software, and we have just released a new feature that allows you to...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 34. `EchoGlitch-Riddle-Decoder`

- URL: https://github.com/kai9987kai/EchoGlitch-Riddle-Decoder
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 12 KB / 2 files
- Features: webapp, html
- Focus: - **Riddle Engine**: A collection of riddles with varying difficulty levels. - **Dynamic Glitch Effects**: Text is "encrypted" with a variety of Unicode character visual effects. - **Gamification**: - **Scoring System...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 35. `EcoSim`

- URL: https://github.com/kai9987kai/EcoSim
- Category: AI Simulations / Games
- Primary language: Python
- Size/File count: 156 KB / 8 files
- Features: webapp, html, python, ai_ml, game
- Focus: EcoSim is a Python-based simulation of an ecosystem with plants and animals, showcasing interactions such as hunting and communication. Animals use machine learning (SGDRegressor) to optimize their movements and inter...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 36. `Ecosim4-HIVEMIND`

- URL: https://github.com/kai9987kai/Ecosim4-HIVEMIND
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 41 KB / 5 files
- Features: webapp, html, ai_ml, game
- Focus: **EcoSim4-HIVEMIND** is a single-file, browser-based ecosystem + neuroevolution sandbox built around a “hivemind” style learning loop: agents run lightweight neural policies in real time, reproduce with mutation, and ...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 37. `Ecosystem-Simulator-with-Experimental-Adaptive-WIP`

- URL: https://github.com/kai9987kai/Ecosystem-Simulator-with-Experimental-Adaptive-WIP
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 14 KB / 1 files
- Features: webapp, html
- Focus: This repository hosts an advanced, experimental ecosystem simulator that integrates adaptive AI features including transformer-based agent brains, meta-evolution, dynamic learning rate adjustments, and emergency memor...
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 38. `EcoWorld-Simulation`

- URL: https://github.com/kai9987kai/EcoWorld-Simulation
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 2 KB / 1 files
- Features: webapp, html
- Focus: EcoWorld is a canvas-based simulation of nature vs. industrialization. Adjust trees, factories, and fish to see real-time environmental effects. Features dynamic landscapes, air quality metrics, and a speed toggle. Id...
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 39. `Email-Bomber`

- URL: https://github.com/kai9987kai/Email-Bomber
- Category: Misc / Experiments
- Primary language: Java
- Size/File count: 832 KB / 126 files
- Features: None
- Focus: Email Bomber, spams large amounts of emails to another email
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 40. `EnviroGen`

- URL: https://github.com/kai9987kai/EnviroGen
- Category: Games / Simulations
- Primary language: Python
- Size/File count: 27 KB / 9 files
- Features: python, game, desktop_gui, ci
- Focus: EnviroGen is a Python package that allows users to create 2D environments with bots, obstacles, and plants using a Tkinter canvas. The bots can move around the environment, interact with obstacles and plants, and have...
- Strengths: README/doc present; CI workflow configured
- Risks: no obvious tests
- Opportunities: headless simulation test harness for gameplay logic

### 41. `Envo-agent-os`

- URL: https://github.com/kai9987kai/Envo-agent-os
- Category: AI Simulations / Games
- Primary language: Python
- Size/File count: 30 KB / 10 files
- Features: python, ai_ml, game
- Focus: A **minimal custom OS project** containing a **bootloader**, a **graphical kernel simulation**, and a **Python build tool** that generates **bootable media** (floppy image + ISO). :contentReference[oaicite:0]{index=0}...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 42. `evolution-simulator`

- URL: https://github.com/kai9987kai/evolution-simulator
- Category: AI / ML Tools
- Primary language: Python
- Size/File count: 39 KB / 5 files
- Features: webapp, html, python, ai_ml
- Focus: A basic script for bots passing on genes represented as numbers in array (the genes don't do anything but hopfully will soon) Comabt system in all even the blender version except the latest beta atm
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 43. `EvolvedNeuralNetworks`

- URL: https://github.com/kai9987kai/EvolvedNeuralNetworks
- Category: AI / ML Tools
- Primary language: Python
- Size/File count: 12 KB / 6 files
- Features: python, ai_ml
- Focus: In this project, we combine the concepts of neural networks and genetic algorithms. Utilizing Python's NeuroEvolution of Augmenting Topologies (NEAT) package, we generate and evolve simple neural networks. These netwo...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; standardize packaging with pyproject.toml

### 44. `File-Manager`

- URL: https://github.com/kai9987kai/File-Manager
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 85 KB / 25 files
- Features: python, desktop_gui
- Focus: A **basic file manager in Python with a GUI**. :contentReference[oaicite:0]{index=0} This repo contains a few iterations of the same idea (original / experimental / current) so you can compare approaches and keep a “w...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 45. `File-SorterAgent`

- URL: https://github.com/kai9987kai/File-SorterAgent
- Category: AI / ML Tools
- Primary language: Python
- Size/File count: 25 KB / 4 files
- Features: python, ai_ml
- Focus: The **File-SorterAgent** is an advanced reinforcement learning-based system designed to automatically organize files in a directory. It uses a neural network to decide which folder a file should be moved to (e.g., `do...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; standardize packaging with pyproject.toml

### 46. `FlappyBird-Unity`

- URL: https://github.com/kai9987kai/FlappyBird-Unity
- Category: Game / Unity
- Primary language: C#
- Size/File count: 2174 KB / 60 files
- Features: game, unity
- Focus: FlappyBird-Unity
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic

### 47. `fluid-dynamics-bot-evolution`

- URL: https://github.com/kai9987kai/fluid-dynamics-bot-evolution
- Category: AI Simulations / Games
- Primary language: Python
- Size/File count: 40 KB / 10 files
- Features: python, ai_ml, game, desktop_gui
- Focus: This repository contains an implementation of a fluid dynamics bot that uses neural networks and a genetic algorithm to evolve its shape and navigate through obstacles. The bot's shape is dynamically adjusted based on...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 48. `Font-Lab-2`

- URL: https://github.com/kai9987kai/Font-Lab-2
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 21 KB / 2 files
- Features: webapp, html
- Focus: Welcome to the Enhanced Typography Playground! This interactive web tool allows you to customize and preview various typography styles for your text, including font, size, weight, style, color, animation, and more. Yo...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 49. `FontLab`

- URL: https://github.com/kai9987kai/FontLab
- Category: Misc / Experiments
- Primary language: Visual Basic
- Size/File count: 22 KB / 18 files
- Features: None
- Focus: Font Lab is an environment for testing fonts
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 50. `FractalAgents`

- URL: https://github.com/kai9987kai/FractalAgents
- Category: AI / ML Tools
- Primary language: HTML
- Size/File count: 59 KB / 5 files
- Features: webapp, html, ai_ml
- Focus: A cosmic fractal explorers club using AI-driven camera angles and color palettes for stunning WebGL fractal visuals, with dual RL agents shaping the evolving scene.
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 51. `GCSE-NEA-2020`

- URL: https://github.com/kai9987kai/GCSE-NEA-2020
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 28 KB / 10 files
- Features: python
- Focus: This was my GCSE NEA i wrote in 2020 due to covid we had to do the same tasks as the years before did sadly
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 52. `GDPL-Assignment-2`

- URL: https://github.com/kai9987kai/GDPL-Assignment-2
- Category: Game / Unity
- Primary language: C#
- Size/File count: 2355 KB / 467 files
- Features: unity
- Focus: GDPL-Assignment-2 at uni in year 1
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 53. `GearboxSimulations`

- URL: https://github.com/kai9987kai/GearboxSimulations
- Category: Games / Simulations
- Primary language: HTML
- Size/File count: 57 KB / 9 files
- Features: webapp, html, game
- Focus: A small collection of **browser-based gearbox / powertrain simulation pages** (single-file `.html` documents) focused on **geartrain kinematics, torque multiplication, stress/thermal “bloom” visualization concepts, an...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic; deploy preset + environment template

### 54. `gemminispark`

- URL: https://github.com/kai9987kai/gemminispark
- Category: Web Apps / Generators
- Primary language: TypeScript
- Size/File count: 14 KB / 13 files
- Features: webapp, html, typescript
- Focus: <div align="center"> <img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" /> </div> This contains everything you need to run your app loca...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 55. `GLSL-Sandox-Scripts`

- URL: https://github.com/kai9987kai/GLSL-Sandox-Scripts
- Category: Misc / Experiments
- Primary language: GLSL
- Size/File count: 20 KB / 5 files
- Features: None
- Focus: Here are all my GLSL animation scripts
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 56. `Guessing-Game`

- URL: https://github.com/kai9987kai/Guessing-Game
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 4 KB / 2 files
- Features: python
- Focus: Basic guessing game don't know why I upload this but feel free to add to the code
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 57. `GutLining-Sim`

- URL: https://github.com/kai9987kai/GutLining-Sim
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 19 KB / 3 files
- Features: webapp, html, ai_ml, game
- Focus: **GutLining-Sim** is a **single-file, browser-based educational simulation** that models a few *mechanistic drivers* that can contribute to **GI mucosal irritation**, then converts them into a single illustrative **“r...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 58. `Hardware-tools`

- URL: https://github.com/kai9987kai/Hardware-tools
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 14 KB / 4 files
- Features: webapp, html
- Focus: A tiny, **single-file** static web project (`index.html`) intended to act as a **hardware tools hub** (a lightweight page you can open locally or host on GitHub Pages). :contentReference[oaicite:0]{index=0} At the mom...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 59. `Historical-Timeline-Tool`

- URL: https://github.com/kai9987kai/Historical-Timeline-Tool
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 14 KB / 2 files
- Features: webapp, html
- Focus: Create and visualize historical timelines with events, dates, descriptions, and icons. - [Description](#description) - [Features](#features) - [Usage](#usage) - [Contributing](#contributing) - [License](#license) This...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 60. `Html-Compactor`

- URL: https://github.com/kai9987kai/Html-Compactor
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 24 KB / 5 files
- Features: webapp, html
- Focus: **Html-Compactor** is a single-file, browser-first concept for an **HTML compression / optimization tool** branded as **“Quantum HTML Optimizer.”** The page is designed around a simple workflow: 1. Paste **Input HTML*...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 61. `HtmlGridLayout-Generator`

- URL: https://github.com/kai9987kai/HtmlGridLayout-Generator
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 39 KB / 7 files
- Features: webapp, html
- Focus: Interactive Grid Builder is a simple web tool that allows users to create and customize a grid-based layout with draggable elements. It is built using HTML, CSS, and JavaScript, providing a user-friendly interface for...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 62. `Hybrid-Super-Brain-Simulation`

- URL: https://github.com/kai9987kai/Hybrid-Super-Brain-Simulation
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 37 KB / 5 files
- Features: webapp, html, ai_ml, game
- Focus: **Hybrid-Super-Brain-Simulation** is a browser-based prototype UI for a **hybrid neuro–quantum–biological simulation** that combines: - a **Gene Regulatory Network (GRN)** layer (gene interactions + environmental nois...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 63. `Hyper-EvoBlob-Simulation`

- URL: https://github.com/kai9987kai/Hyper-EvoBlob-Simulation
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 19 KB / 2 files
- Features: webapp, html, ai_ml, game
- Focus: A **single-file, browser-based simulation** packaged as `index.html`. The repo currently contains **only** that file (HTML/JS) and no additional build tooling, dependencies, or project metadata. :contentReference[oaic...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 64. `Infinite-scroll`

- URL: https://github.com/kai9987kai/Infinite-scroll
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 22 KB / 5 files
- Features: webapp, html
- Focus: Website where you can scroll down or right infinitely
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 65. `Interactive-gearbox-diagram-template`

- URL: https://github.com/kai9987kai/Interactive-gearbox-diagram-template
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 20 KB / 5 files
- Features: webapp, html
- Focus: A **single-file HTML template** for presenting an **interactive gearbox diagram** in the browser. This repository is intentionally minimal: it currently contains **one HTML file** — `lift_gearbox_diagram.html` — and G...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 66. `Interactive-heart`

- URL: https://github.com/kai9987kai/Interactive-heart
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 20 KB / 3 files
- Features: webapp, html
- Focus: A lightweight **browser-based interactive heart project** built as a minimal static site. This repo ships as a simple HTML app and uses the **Gemini API** for AI-powered explanations/interaction. > ⚠️ You must provide...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 67. `interactive-pdf`

- URL: https://github.com/kai9987kai/interactive-pdf
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 5891 KB / 16 files
- Features: python
- Focus: A Python-based project for generating and/or processing **interactive PDF workflows** using a clean input/output folder structure. This repository is organized to keep source PDFs, processing scripts, and generated PD...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 68. `invisible-zalgo`

- URL: https://github.com/kai9987kai/invisible-zalgo
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 11 KB / 4 files
- Features: webapp, html
- Focus: Small browser-based experiments for invisible Unicode text effects and hidden-message encoding. This project contains two standalone HTML tools: - `index.html`: a "Zalgo" generator that uses only `U+2800` (Braille Pat...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 69. `IP-details`

- URL: https://github.com/kai9987kai/IP-details
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 35 KB / 10 files
- Features: python
- Focus: **IP-details** is a small Python script that fetches metadata about **your public IP address** (e.g., location, IP type, and related fields) using the **ipstack** IP geolocation API. - Queries ipstack’s API to retriev...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 70. `IPfinder`

- URL: https://github.com/kai9987kai/IPfinder
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 32 KB / 5 files
- Features: python
- Focus: IP finder gets the IP of any website.
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 71. `kai9987kai`

- URL: https://github.com/kai9987kai/kai9987kai
- Category: AI Simulations / Games
- Primary language: Unknown
- Size/File count: 57 KB / 2 files
- Features: ai_ml, game
- Focus: <p align="center"> Kai9987kai, a passionate and dedicated developer called Kai Piper, has been actively contributing to the open-source community on GitHub, with a keen interest in Python and Java. </p> - 👋 Hi, I’m Ka...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 72. `kai9987kai.github.io`

- URL: https://github.com/kai9987kai/kai9987kai.github.io
- Category: Game / Unity
- Primary language: HTML
- Size/File count: 469408 KB / 1123 files
- Features: webapp, html, javascript, unity, tests, ci
- Focus: This repository contains my personal website hosted on GitHub Pages. It’s a mixed-purpose site that I use to: - Showcase **projects** and interactive web experiments (HTML/CSS/JS) - Link **social accounts** (YouTube c...
- Strengths: README/doc present; tests present; CI workflow configured
- Risks: No major automated flags
- Opportunities: deploy preset + environment template

### 73. `Kai_Piper_App_Project`

- URL: https://github.com/kai9987kai/Kai_Piper_App_Project
- Category: Misc / Experiments
- Primary language: Kotlin
- Size/File count: 2424 KB / 69 files
- Features: tests
- Focus: This app is made to highlight my projects and to test ideas and app features
- Strengths: README/doc present; tests present
- Risks: No major automated flags
- Opportunities: add GitHub Actions for smoke tests/build

### 74. `keycoach`

- URL: https://github.com/kai9987kai/keycoach
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 56 KB / 5 files
- Features: webapp, html
- Focus: Want to improve your typing? Use this, it will help!
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 75. `Labyrinth`

- URL: https://github.com/kai9987kai/Labyrinth
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 74 KB / 5 files
- Features: webapp, html, ai_ml, game
- Focus: A **single-file** (HTML + CSS + JS) reflective maze experience that combines **procedural maze navigation** with **journaling prompts** and an optional **LLM “coach”**. It supports **seeded, shareable mazes**, **fog-o...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 76. `Largest-Folder-Finder`

- URL: https://github.com/kai9987kai/Largest-Folder-Finder
- Category: Misc / Experiments
- Primary language: Visual Basic .NET
- Size/File count: 841 KB / 146 files
- Features: tests
- Focus: Must be ran as admin
- Strengths: README/doc present; tests present
- Risks: No major automated flags
- Opportunities: add GitHub Actions for smoke tests/build

### 77. `learning-neual-net-with-tesorflow`

- URL: https://github.com/kai9987kai/learning-neual-net-with-tesorflow
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 2 KB / 1 files
- Features: python
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 78. `logo-generator`

- URL: https://github.com/kai9987kai/logo-generator
- Category: Desktop Utilities / GUI
- Primary language: C
- Size/File count: 4629 KB / 43 files
- Features: python, desktop_gui
- Focus: A powerful Python-based tool for generating modern, harmonious logos with a sleek, dark-themed GUI. The easiest way to run the application is using the provided batch file: 1. Open the project folder. 2. Double-click ...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 79. `lower-upper-case-converter`

- URL: https://github.com/kai9987kai/lower-upper-case-converter
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 10 KB / 4 files
- Features: python
- Focus: Text character and converts text to upper and lower case
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 80. `MalDuinoScripts`

- URL: https://github.com/kai9987kai/MalDuinoScripts
- Category: Misc / Experiments
- Primary language: C++
- Size/File count: 27 KB / 16 files
- Features: None
- Focus: A collection of scripts for the MalDuino lite
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 81. `Map-Builder`

- URL: https://github.com/kai9987kai/Map-Builder
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 84 KB / 5 files
- Features: webapp, html
- Focus: **Map-Builder** is a lightweight, single-file, browser-first **2D map/canvas editor** concept. The repository currently contains one file (`index.html`) that defines the intended tool surface for drawing, layout, laye...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 82. `Marsgearboxsim`

- URL: https://github.com/kai9987kai/Marsgearboxsim
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 42 KB / 5 files
- Features: webapp, html
- Focus: Educational **lumped-parameter** simulator for a **single spur-gear stage** operating in a **Mars-like environment** (low-pressure CO₂, weak convection, cold ambient, dust). The goal is a *plausible* “systems feel” (l...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 83. `Maths-PuzzelUltra`

- URL: https://github.com/kai9987kai/Maths-PuzzelUltra
- Category: Games / Simulations
- Primary language: Python
- Size/File count: 19 KB / 10 files
- Features: python, game
- Focus: Maths-PuzzelUltra Maths-PuzzelUltra is an arcade-style math puzzle game built with Pygame. Solve generated problems under different modes, build combo multipliers, fill a power gauge (time-slow), and reveal a picture ...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic; standardize packaging with pyproject.toml

### 84. `Maya-Scripts`

- URL: https://github.com/kai9987kai/Maya-Scripts
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 105 KB / 27 files
- Features: python
- Focus: Some Maya scripts for useful stuff or some novel stuff
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 85. `MazeGen-solve-ai`

- URL: https://github.com/kai9987kai/MazeGen-solve-ai
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 7 KB / 4 files
- Features: webapp, html, python, desktop_gui
- Focus: This script uses the tkinter module, which is a built-in module for creating graphical user interfaces, to generate a maze and display it in a canvas. The MazeGenerator class uses a depth-first search algorithm to gen...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template; standardize packaging with pyproject.toml

### 86. `MCPCMS`

- URL: https://github.com/kai9987kai/MCPCMS
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 21 KB / 5 files
- Features: webapp, html
- Focus: medical chest and pneumonia concept medical assessor
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 87. `Media-PiP-Enhancer`

- URL: https://github.com/kai9987kai/Media-PiP-Enhancer
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 2 KB / 1 files
- Features: webapp, html
- Focus: "An advanced Picture-in-Picture (PiP) web tool that supports images and videos. Features customizable PiP settings, including size, border, and controls."  This succinct description and name should give potential user...
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 88. `Microsoft-Small-Basic-Examples`

- URL: https://github.com/kai9987kai/Microsoft-Small-Basic-Examples
- Category: Misc / Experiments
- Primary language: Unknown
- Size/File count: 10 KB / 6 files
- Features: None
- Focus: Microsoft Smal Basic is very simple programming language I have decided to put a few examples on this page to help beginners
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 89. `MindScapes`

- URL: https://github.com/kai9987kai/MindScapes
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 29 KB / 5 files
- Features: webapp, html
- Focus: **MindScapes** is an experimental web project exploring *cognitive / psychological “mindscape”* themes through **AI-inspired worldbuilding**, **procedural textures**, and **dynamic visual layers**—currently starting a...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 90. `Multi-Agent-Ontoverse`

- URL: https://github.com/kai9987kai/Multi-Agent-Ontoverse
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 100 KB / 3 files
- Features: webapp, html
- Focus: A browser-based multi-agent environment that demonstrates hierarchical tasks, synergy requirements, day/night cycles, and a commander RL agent—powered by TensorFlow.js Dueling DQN. Inspired by “Ontoverse”-style knowle...
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 91. `Multidisciplinary-Python-Data-Analysis`

- URL: https://github.com/kai9987kai/Multidisciplinary-Python-Data-Analysis
- Category: AI / ML Tools
- Primary language: Python
- Size/File count: 2 KB / 2 files
- Features: python, ai_ml
- Focus: A Python script demonstrating various data analysis techniques including numerical computation, data manipulation, machine learning, visualization, and natural language processing. Numerical computation (numpy): It us...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; standardize packaging with pyproject.toml

### 92. `MultiModalFusion`

- URL: https://github.com/kai9987kai/MultiModalFusion
- Category: AI / ML Tools
- Primary language: HTML
- Size/File count: 40 KB / 5 files
- Features: webapp, html, ai_ml
- Focus: A minimal, browser-first **multimodal ML control panel** scaffold that groups two training/inference workflows in one place: - **Language model workflow** (pre-train, fine-tune, generate, save/load) - **Tabular ML wor...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 93. `Neon-Bio-OS`

- URL: https://github.com/kai9987kai/Neon-Bio-OS
- Category: AI Simulations / Games
- Primary language: JavaScript
- Size/File count: 33 KB / 11 files
- Features: webapp, html, ai_ml, game, desktop_gui, javascript
- Focus: **Neon Bio-OS** is a browser-based experimental interface concept that blends **neon cyber aesthetics** with **bio/ecosystem-inspired system design**. > “Based on your extensive history of building advanced UI mockups...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 94. `Neural-network-animation`

- URL: https://github.com/kai9987kai/Neural-network-animation
- Category: AI / ML Tools
- Primary language: HTML
- Size/File count: 15 KB / 3 files
- Features: webapp, html, ai_ml
- Focus: Neural network-based animation
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 95. `Neural-Network-Playground`

- URL: https://github.com/kai9987kai/Neural-Network-Playground
- Category: AI / ML Tools
- Primary language: HTML
- Size/File count: 12 KB / 5 files
- Features: webapp, html, ai_ml
- Focus: **Neural-Network-Playground** is a lightweight, browser-first scaffold for an **interactive neural network architecture builder**. The repo currently contains a single `index.html` file that outlines the intended UI/f...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 96. `Neuro-Visual-Synth-Lab`

- URL: https://github.com/kai9987kai/Neuro-Visual-Synth-Lab
- Category: Games / Simulations
- Primary language: HTML
- Size/File count: 16 KB / 2 files
- Features: webapp, html, game, desktop_gui
- Focus: **Neuro-Visual-Synth-Lab** is a **single-file, in-browser lab** that combines a **live audio engine** (binaural / isochronic / noise) with multiple **real-time visual modes** (Plasma / Boids / Reaction-Diffusion) plus...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic; deploy preset + environment template

### 97. `NeuroBoxingSimulator`

- URL: https://github.com/kai9987kai/NeuroBoxingSimulator
- Category: AI Simulations / Games
- Primary language: Python
- Size/File count: 2 KB / 2 files
- Features: python, ai_ml, game
- Focus: A Python-based project combining neural networks and game simulation. It creates a simple feedforward neural network and visualizes a boxing match in a plot
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 98. `NeuroDSL-Infinity-Studio`

- URL: https://github.com/kai9987kai/NeuroDSL-Infinity-Studio
- Category: DSL / Compilers
- Primary language: Python
- Size/File count: 49 KB / 13 files
- Features: python, ai_ml, game, desktop_gui, dsl, tests
- Focus: A **GUI-first neural architecture sandbox** that lets you describe modern PyTorch networks in a compact **DSL (Domain Specific Language)**, compile them into a runnable model, **train** (synthetic or CSV), **run infer...
- Strengths: README/doc present; tests present; experiments with DSL/abstraction
- Risks: No major automated flags
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; standardize packaging with pyproject.toml

### 99. `NeuroDSL-Infinity-Studio2`

- URL: https://github.com/kai9987kai/NeuroDSL-Infinity-Studio2
- Category: DSL / Compilers
- Primary language: Python
- Size/File count: 18233 KB / 162 files
- Features: webapp, html, python, ai_ml, game, desktop_gui, javascript, dsl, tests
- Focus: A **GUI-first neural architecture sandbox** that lets you describe modern PyTorch networks in a compact **DSL (Domain Specific Language)**, compile them into a runnable model, **train** (synthetic or CSV), **run infer...
- Strengths: README/doc present; tests present; experiments with DSL/abstraction
- Risks: No major automated flags
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 100. `NeuroEvolution-Simulation`

- URL: https://github.com/kai9987kai/NeuroEvolution-Simulation
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 1 KB / 1 files
- Features: python
- Focus: NeuroEvoSim: Python-based evolutionary neural network simulation. Evolve neural networks in a simulated environment. Sensing, movement, input processing. Adjustable evolution rate, mutation rate. Save/load simulation ...
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 101. `NotificationSpammer`

- URL: https://github.com/kai9987kai/NotificationSpammer
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 29 KB / 7 files
- Features: python
- Focus: NotificationSpammer spams windows notification Modules need: win10toast, webbrowser and keyboard
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 102. `npu_easy`

- URL: https://github.com/kai9987kai/npu_easy
- Category: AI / ML Tools
- Primary language: Python
- Size/File count: 36 KB / 14 files
- Features: python, ai_ml, ci
- Focus: ```bash pip install . ``` `npu-easy` has **zero hard dependencies**. It will install quickly without downloading large packages. To actually use NPU acceleration, you should install the appropriate ONNX Runtime versio...
- Strengths: README/doc present; CI workflow configured; AI/simulation focus
- Risks: no obvious tests
- Opportunities: reproducible experiment config and deterministic seeds

### 103. `Num-Network`

- URL: https://github.com/kai9987kai/Num-Network
- Category: AI / ML Tools
- Primary language: Python
- Size/File count: 24 KB / 6 files
- Features: python, ai_ml, desktop_gui
- Focus: A tiny, **numeric** “chatbot” experiment that uses a **single-neuron neural network (logistic regression)** to map **(user_id, message_id) → response_score**. It trains from a CSV (`chat_data.csv`) and then runs an in...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; standardize packaging with pyproject.toml

### 104. `Octo-browse`

- URL: https://github.com/kai9987kai/Octo-browse
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 87 KB / 5 files
- Features: python, desktop_gui
- Focus: Octo-browse Octo-browse is an experimental, Python + PyQt6/QtWebEngine desktop browser prototype that mixes traditional browsing with “power tools” like tabbed browsing, ad-blocking via request interception, incognito...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 105. `Online-Ender-Slicer`

- URL: https://github.com/kai9987kai/Online-Ender-Slicer
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 20 KB / 2 files
- Features: webapp, html
- Focus: wired web slicer
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 106. `OpenCad`

- URL: https://github.com/kai9987kai/OpenCad
- Category: AI / ML Tools
- Primary language: Python
- Size/File count: 22 KB / 17 files
- Features: python, ai_ml
- Focus: **OpenCad** is an early-stage **Python CAD application / CAD-kernel playground**. The repository currently contains: - `main.py` (entry point) - `src/` (project code) - `.gitattributes` GitHub shows **Python 100%**, t...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; standardize packaging with pyproject.toml

### 107. `OpenDocs`

- URL: https://github.com/kai9987kai/OpenDocs
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 102 KB / 5 files
- Features: webapp, html
- Focus: **OpenDocs** is a minimalist, **single-file HTML** project intended as a starting point for a lightweight documentation page you can open locally or host as a static site. The repository currently contains **one file*...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 108. `OpenSheets`

- URL: https://github.com/kai9987kai/OpenSheets
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 41 KB / 4 files
- Features: webapp, html
- Focus: OpenSheets is an **experimental, single-file web app** (everything lives in `index.html`) that’s intended as a lightweight “spreadsheet-like” playground you can run locally or host as a static site. :contentReference[...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 109. `OpenSkin`

- URL: https://github.com/kai9987kai/OpenSkin
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 4 KB / 1 files
- Features: python
- Focus: A windows os skin customizer allowing you to customize windows
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 110. `OpenSupplement`

- URL: https://github.com/kai9987kai/OpenSupplement
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 42 KB / 5 files
- Features: webapp, html
- Focus: working on new ways to help people navigate the extreme amount of supplements and their benefits
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 111. `Openworld`

- URL: https://github.com/kai9987kai/Openworld
- Category: Game / Unity
- Primary language: Rich Text Format
- Size/File count: 2966278 KB / 4220 files
- Features: game, unity
- Focus: Small Open world environment in unity
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic

### 112. `PC-DANCE`

- URL: https://github.com/kai9987kai/PC-DANCE
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 27 KB / 9 files
- Features: python
- Focus: PC Dance is a project to make the software and hardware of your computer act like its dancing we currently have 2 options. First is a script that makes your keyboard lights flicker. Secondly a script that makes the di...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 113. `petrichor_v3`

- URL: https://github.com/kai9987kai/petrichor_v3
- Category: AI / ML Tools
- Primary language: JavaScript
- Size/File count: 145 KB / 100 files
- Features: webapp, html, ai_ml, desktop_gui, javascript
- Focus: **Petrichor Lab v3** is a stylized, modular *control-surface* concept for building rain/atmosphere “scenes” — an interface spec that reads like a sci-fi synth panel for **environment + weather + psychoacoustics + gene...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 114. `PinBall`

- URL: https://github.com/kai9987kai/PinBall
- Category: Game / Unity
- Primary language: Wolfram Language
- Size/File count: 343 KB / 117 files
- Features: game, unity
- Focus: Unity pinball game
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic

### 115. `pizzapi`

- URL: https://github.com/kai9987kai/pizzapi
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 90 KB / 28 files
- Features: python, desktop_gui, tests
- Focus: pizzapy Disclaimer This is my fork of https://github.com/gamagori/pizzapi It's heavily modified and not well documented, but i'm going to get to that. the below example should work though. sorry! was kind of in a rush...
- Strengths: README/doc present; tests present
- Risks: No major automated flags
- Opportunities: add GitHub Actions for smoke tests/build

### 116. `plant-ecosystem-sim`

- URL: https://github.com/kai9987kai/plant-ecosystem-sim
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 6 KB / 2 files
- Features: webapp, html, ai_ml, game
- Focus: An interactive plant growth simulation built with vanilla JavaScript. Features genetic traits, weather effects, and adjustable speed controls. Grow and harvest plants in a dynamic ecosystem! An interactive plant growt...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 117. `Pong-extra`

- URL: https://github.com/kai9987kai/Pong-extra
- Category: Games / Simulations
- Primary language: JavaScript
- Size/File count: 7 KB / 4 files
- Features: webapp, html, game, desktop_gui, javascript
- Focus: A neon-styled, browser-based Pong variant built with **vanilla HTML5 Canvas + JavaScript**, featuring: - **1 Player vs AI** and **2 Player** modes - **Dash mechanic** (Shift) - **Power-ups** (Multi-ball + Big Paddle) ...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic; deploy preset + environment template

### 118. `ProteinFoldingSimulation`

- URL: https://github.com/kai9987kai/ProteinFoldingSimulation
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 26 KB / 3 files
- Features: webapp, html, ai_ml, game
- Focus: Single-file website with 3D protein visualization, a neural network prioritizer, and folding simulation What This Includes 🧠 Neural Network Priority System 12-input feature extraction from amino acid sequences (hydrop...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 119. `PyCMD`

- URL: https://github.com/kai9987kai/PyCMD
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 12 KB / 4 files
- Features: python
- Focus: A Simple Version Of CMD, Simpler Syntax Etc. Use This In a School, Workplace Etc If you don't have permission to access CMD You have python! You have PyCMD!
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 120. `pygame-tutorials`

- URL: https://github.com/kai9987kai/pygame-tutorials
- Category: Games / Simulations
- Primary language: Python
- Size/File count: 6651 KB / 54 files
- Features: python, game
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic; standardize packaging with pyproject.toml

### 121. `pynocode`

- URL: https://github.com/kai9987kai/pynocode
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 29 KB / 3 files
- Features: webapp, html, desktop_gui
- Focus: Web Drag and drop no code interface for Tkinter
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 122. `PyOBJViewer`

- URL: https://github.com/kai9987kai/PyOBJViewer
- Category: AI Simulations / Games
- Primary language: Python
- Size/File count: 28 KB / 5 files
- Features: python, ai_ml, game, desktop_gui
- Focus: A minimal **Wavefront `.obj` model viewer** written in Python. The repo is intentionally lightweight and currently centers around a single entrypoint script (`main.py`) plus standard community/security files. :content...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 123. `Python`

- URL: https://github.com/kai9987kai/Python
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 4387 KB / 200 files
- Features: python, desktop_gui, tests
- Focus: Here is more detailed information about scripts I have written. I do not consider myself a programmer; I create these little programs as experiments to play with the language, or to solve problems for myself. I would ...
- Strengths: README/doc present; tests present
- Risks: No major automated flags
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 124. `Python-Mass-File-Maker`

- URL: https://github.com/kai9987kai/Python-Mass-File-Maker
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 7 KB / 3 files
- Features: python
- Focus: Do not use this maliciously and make sure to only use on a vm Growth rate: Batch 5: ~1.25 MB total Batch 10: ~1.3 GB total Batch 15: ~1.3 TB total Batch 20: ~1.3 PB total (if you had that much space!) In practical ter...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 125. `Quan-Dream-Weaver`

- URL: https://github.com/kai9987kai/Quan-Dream-Weaver
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 18 KB / 4 files
- Features: webapp, html
- Focus: **Quan-Dream-Weaver** is a lightweight, single-file browser experiment built as a standalone `index.html` (HTML-only repo) — ideal for quick generative / “dream-like” interactive visuals, prototypes, and shareable web...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 126. `Quantum-Fractal-Nexus`

- URL: https://github.com/kai9987kai/Quantum-Fractal-Nexus
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 28 KB / 1 files
- Features: webapp, html
- Focus: A real-time hybrid simulation that merges fractal rendering with “quantum” particle behavior. It features:
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 127. `Quantum-Nexus-Advanced-Neural-Interface-Simulation`

- URL: https://github.com/kai9987kai/Quantum-Nexus-Advanced-Neural-Interface-Simulation
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 35 KB / 2 files
- Features: webapp, html
- Focus: A comprehensive, interactive simulation of a quantum neural interface built with HTML, CSS, and JavaScript. Features include real-time simulation state overlays, voice command control, dynamic node animations, ambient...
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 128. `QuantumBot`

- URL: https://github.com/kai9987kai/QuantumBot
- Category: AI Simulations / Games
- Primary language: Python
- Size/File count: 22 KB / 5 files
- Features: python, ai_ml, game, desktop_gui
- Focus: QuantumBot is a unique simulation of bots controlled by a quantum neural network. Using Qiskit for quantum computations, PyTorch for neural networks, and Tkinter for the GUI, it showcases the intersection of quantum c...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 129. `QuantumParticleSimulator`

- URL: https://github.com/kai9987kai/QuantumParticleSimulator
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 9 KB / 2 files
- Features: webapp, html
- Focus: Quantum Mechanics Visualization: Wave-particle duality, entanglement networks, and charge interactions Interactive Fields: Click to spawn quantum potential wells Hyperspace Projection: 4D Klein bottle topology renderi...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 130. `Race-ai`

- URL: https://github.com/kai9987kai/Race-ai
- Category: AI / ML Tools
- Primary language: Python
- Size/File count: 35 KB / 3 files
- Features: python, ai_ml
- Focus: **Race-ai** is an experimental Python project exploring an AI agent that learns to get past obstacles and reach the finish as fast as possible. > Repo tagline: “Ai learning to get past obsticles to the finsih in the f...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; standardize packaging with pyproject.toml

### 131. `Random-max-scripts`

- URL: https://github.com/kai9987kai/Random-max-scripts
- Category: Misc / Experiments
- Primary language: MAXScript
- Size/File count: 5 KB / 3 files
- Features: None
- Focus: random 3ds max scripts
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 132. `Random-Youtube-Video`

- URL: https://github.com/kai9987kai/Random-Youtube-Video
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 31 KB / 6 files
- Features: python
- Focus: Random Youtube Video form a google sheets document using the Google sheets api
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 133. `RapidBackgroundChanger`

- URL: https://github.com/kai9987kai/RapidBackgroundChanger
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 26 KB / 5 files
- Features: python
- Focus: Rapid Background Changer rapidly changes your windows background. python modules needed: pywin32 and keyboard
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 134. `Registerymanager`

- URL: https://github.com/kai9987kai/Registerymanager
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 76 KB / 37 files
- Features: python, desktop_gui, tests
- Focus: A Windows-only Registry Manager written in Python, with a **CustomTkinter** GUI for browsing, editing, backing up, and applying “preset” tweaks to common Windows settings. > ⚠️ **Warning:** Editing the Windows Registr...
- Strengths: README/doc present; tests present
- Risks: No major automated flags
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 135. `Rendering-experiments`

- URL: https://github.com/kai9987kai/Rendering-experiments
- Category: AI / ML Tools
- Primary language: HTML
- Size/File count: 29 KB / 9 files
- Features: webapp, html, ai_ml
- Focus: A grab-bag of **single-file HTML rendering prototypes** exploring “neural-ish” visual ideas in the browser: procedural deformation, toy TensorFlow.js modulation, texture generation experiments, and a physics/collision...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 136. `Rex-ATSLM`

- URL: https://github.com/kai9987kai/Rex-ATSLM
- Category: AI / ML Tools
- Primary language: HTML
- Size/File count: 169 KB / 5 files
- Features: webapp, html, ai_ml
- Focus: Active training small language model
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 137. `Ribbon-Bar-Template`

- URL: https://github.com/kai9987kai/Ribbon-Bar-Template
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 29 KB / 5 files
- Features: webapp, html
- Focus: Html Ribbon-Bar-Template Public
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 138. `Roblox-Scripts`

- URL: https://github.com/kai9987kai/Roblox-Scripts
- Category: AI Simulations / Games
- Primary language: Lua
- Size/File count: 28 KB / 9 files
- Features: ai_ml, game
- Focus: Script1 = script that implements parallel script execution with conflict resolution and live error correction. This system uses adaptive sandboxing and heuristic-based patching Script2 = script with advanced categoriz...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 139. `Rock-Paper-Scissors-with-GUI-NEA-MOCK`

- URL: https://github.com/kai9987kai/Rock-Paper-Scissors-with-GUI-NEA-MOCK
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 37 KB / 8 files
- Features: python, desktop_gui
- Focus: We had to do a NEA mock in computer science so I made this...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 140. `Rotary-Generator`

- URL: https://github.com/kai9987kai/Rotary-Generator
- Category: AI Simulations / Games
- Primary language: HTML
- Size/File count: 87 KB / 6 files
- Features: webapp, html, ai_ml, game
- Focus: **Rotary-Generator** is a lightweight, browser-first prototype/spec for an **interactive rotary generator + gearbox design environment**. It focuses on capturing the full parameter surface you’d expect in an engineeri...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; headless simulation test harness for gameplay logic

### 141. `scribble-diffusion`

- URL: https://github.com/kai9987kai/scribble-diffusion
- Category: AI / ML Tools
- Primary language: Unknown
- Size/File count: 1410 KB / 41 files
- Features: webapp, ai_ml, javascript, tests, ci
- Focus: Try it out at [scribblediffusion.com](https://scribblediffusion.com) This app is powered by: 🚀 [Replicate](https://replicate.com/?utm_source=project&utm_campaign=scribblediffusion), a platform for running machine lear...
- Strengths: README/doc present; tests present; CI workflow configured
- Risks: No major automated flags
- Opportunities: reproducible experiment config and deterministic seeds; deploy preset + environment template

### 142. `SEO-optimisedpage`

- URL: https://github.com/kai9987kai/SEO-optimisedpage
- Category: Games / Simulations
- Primary language: HTML
- Size/File count: 14 KB / 2 files
- Features: webapp, html, game
- Focus: A **single-file, SEO-focused landing page** (currently `index.html`) for a creative technologist profile/portfolio. The repo description frames the page around **Kai William Piper (Bournemouth)** and highlights 3D, Un...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic; deploy preset + environment template

### 143. `Shepard-Tone-Beat-and-White-Noise-Generator`

- URL: https://github.com/kai9987kai/Shepard-Tone-Beat-and-White-Noise-Generator
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 9 KB / 2 files
- Features: webapp, html
- Focus: This tool generates a Shepard tone, a basic 4/4 beat, and white noise. Use the controls to adjust the volume, speed, and base frequency of the Shepard tone, the volume and tempo of the beat, and the volume of the whit...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 144. `single-file-lab-studio`

- URL: https://github.com/kai9987kai/single-file-lab-studio
- Category: Games / Simulations
- Primary language: TypeScript
- Size/File count: 15514 KB / 10768 files
- Features: webapp, html, python, game, desktop_gui, javascript, typescript, tests
- Focus: **Generate self-contained HTML/CSS/JS single-file apps with live preview, preset management, and one-click publish-readiness.** - **GUI Sidebar**: A dedicated **Lab Explorer** view in the Activity Bar. - **Template Ga...
- Strengths: README/doc present; tests present
- Risks: No major automated flags
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 145. `StarWarsDataBank`

- URL: https://github.com/kai9987kai/StarWarsDataBank
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 24 KB / 5 files
- Features: python
- Focus: Star Wars Data Bank uses the SWAPI api to get data about a randomly picked star wars character
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 146. `StegoBidi-Invisible-Text`

- URL: https://github.com/kai9987kai/StegoBidi-Invisible-Text
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 6 KB / 1 files
- Features: webapp, html
- Focus: A web-based tool that hides and retrieves secret messages using zero‑width characters, with optional cover‑text embedding for seamless steganography.
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 147. `Stick-Figure`

- URL: https://github.com/kai9987kai/Stick-Figure
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 400 KB / 27 files
- Features: webapp, html
- Focus: A browser-based **stick figure rig lab** focused on animation systems: **IK**, **keyframes**, clip playback, terrain interaction, balance/COM tooling, and simple “beat-reactive” automation hooks. ([GitHub][1]) * **Rig...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 148. `Subscription-Reminder`

- URL: https://github.com/kai9987kai/Subscription-Reminder
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 16 KB / 1 files
- Features: webapp, html
- Focus: This web page allows you to add subscriptions and set reminders for them. Enter the name and due date of your subscription, and you'll receive a browser notification one day before it's due. You can also delete subscr...
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 149. `Supermix_27`

- URL: https://github.com/kai9987kai/Supermix_27
- Category: AI / ML Tools
- Primary language: Python
- Size/File count: 14522 KB / 135 files
- Features: webapp, html, python, ai_ml, javascript
- Focus: Supermix_27 is a structured AI/chat project repository designed for GitHub publishing, with a clear split between source code, runtime assets, and web deployment bundles. It includes open-source training and dataset-b...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 150. `System-Tray-Filler`

- URL: https://github.com/kai9987kai/System-Tray-Filler
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 48 KB / 6 files
- Features: python
- Focus: Welcome to the System Tray Filler! This software Fills the system tray!!!! The longer you run this program the more system resources it users so be careful!!
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 151. `Unicode-Audio-Visualizer`

- URL: https://github.com/kai9987kai/Unicode-Audio-Visualizer
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 1 KB / 1 files
- Features: webapp, html
- Focus: "Web-based audio visualizer that uses Unicode characters to represent audio frequencies. Built with HTML, CSS, and Web Audio API. Ideal for quick audio analysis and visualization."  Feel free to use or modify these fo...
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 152. `UnityScripts`

- URL: https://github.com/kai9987kai/UnityScripts
- Category: Game / Unity
- Primary language: C#
- Size/File count: 37 KB / 12 files
- Features: game
- Focus: some cool unity scripts
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic

### 153. `UnityShaders`

- URL: https://github.com/kai9987kai/UnityShaders
- Category: Game / Unity
- Primary language: ShaderLab
- Size/File count: 85 KB / 7 files
- Features: game
- Focus: Collection of my own unity shaders created in unity shader graph and exported to code
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic

### 154. `UpperLowerCaseConverter-GUI`

- URL: https://github.com/kai9987kai/UpperLowerCaseConverter-GUI
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 19 KB / 6 files
- Features: python, desktop_gui
- Focus: upper and lower case text converter with a GUI
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 155. `Visual-Sequence-Synthesizer`

- URL: https://github.com/kai9987kai/Visual-Sequence-Synthesizer
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 15 KB / 5 files
- Features: webapp, html
- Focus: A tiny, static **browser-based “visual sequence synthesizer”** prototype built as a single-page project (currently just an `index.html` in the repo). :contentReference[oaicite:0]{index=0} At the moment, the UI copy in...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 156. `Voronoi-Gradient-Noise-Generator`

- URL: https://github.com/kai9987kai/Voronoi-Gradient-Noise-Generator
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 2 KB / 1 files
- Features: python
- Focus: A Python-based image generator that replicates Voronoi and gradient noise effects, inspired by a Unity shader.
- Strengths: No notable automated strengths detected
- Risks: missing or minimal README; no obvious tests; very small repo size (possibly placeholder or early stub)
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 157. `Warcraft-camera-unity`

- URL: https://github.com/kai9987kai/Warcraft-camera-unity
- Category: Game / Unity
- Primary language: C#
- Size/File count: 22 KB / 5 files
- Features: game, desktop_gui
- Focus: The `WarcraftCameraController` is a Unity script designed to create a camera system similar to the one used in Warcraft games. It locks the camera at a specific angle, mimicking the isometric view typical in such game...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic

### 158. `web-cam-withgui`

- URL: https://github.com/kai9987kai/web-cam-withgui
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 15 KB / 6 files
- Features: python, desktop_gui
- Focus: A desktop webcam application with a modern Tkinter GUI and an OpenCV capture pipeline. It supports multi-camera selection, live filters, optional face tracking overlays, snapshots, and video recording. Repo contains: ...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 159. `Web-flappybird`

- URL: https://github.com/kai9987kai/Web-flappybird
- Category: Games / Simulations
- Primary language: HTML
- Size/File count: 27 KB / 5 files
- Features: webapp, html, game
- Focus: Online flappy bird game
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; headless simulation test harness for gameplay logic; deploy preset + environment template

### 160. `Web-hardwaremonitor`

- URL: https://github.com/kai9987kai/Web-hardwaremonitor
- Category: AI / ML Tools
- Primary language: HTML
- Size/File count: 80 KB / 3 files
- Features: webapp, html, ai_ml
- Focus: The Task Manager Dashboard is a web-based tool that provides real-time information about system performance and various web-related metrics. It's designed to give users a comprehensive view of their system's status, a...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 161. `webmaya`

- URL: https://github.com/kai9987kai/webmaya
- Category: AI / ML Tools
- Primary language: HTML
- Size/File count: 32 KB / 5 files
- Features: webapp, html, ai_ml
- Focus: **WebMaya** is an experimental, single-file, browser-based 3D mini-studio inspired by DCC workflows (Maya/Blender-style layout). It runs entirely client-side and focuses on fast iteration: create primitives, transform...
- Strengths: README/doc present; AI/simulation focus
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; reproducible experiment config and deterministic seeds; deploy preset + environment template

### 162. `WebNotes`

- URL: https://github.com/kai9987kai/WebNotes
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 26 KB / 6 files
- Features: webapp, html
- Focus: **WebNotes** is a lightweight, **single-file** browser notes app (“Advanced Enhanced Notes Application”) designed for quick note-taking with basic rich-text controls and live text stats. It runs entirely in the browse...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 163. `weird-image-filemaker`

- URL: https://github.com/kai9987kai/weird-image-filemaker
- Category: Python Utilities / Scripts
- Primary language: Python
- Size/File count: 20 KB / 9 files
- Features: python
- Focus: A Python-based toolkit for generating, mutating, hardening, and packaging collections of “weird” images. This repository provides command-line utilities and mutation pipelines to programmatically transform source imag...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 164. `What-browser-am-I-using`

- URL: https://github.com/kai9987kai/What-browser-am-I-using
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 8 KB / 3 files
- Features: webapp, html
- Focus: Browsers include Google Chrome, Firefox, Safari, and Internet Explorer.
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 165. `Wifi-Manager`

- URL: https://github.com/kai9987kai/Wifi-Manager
- Category: Desktop Utilities / GUI
- Primary language: Python
- Size/File count: 8 KB / 3 files
- Features: python, desktop_gui
- Focus: A lightweight **Windows Wi-Fi manager GUI** built with **Python + CustomTkinter**, backed by the native `netsh wlan` command set. It’s split into: - `wifi_backend.py` — a small backend wrapper around `netsh` (scan, co...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; standardize packaging with pyproject.toml

### 166. `Win_Automations`

- URL: https://github.com/kai9987kai/Win_Automations
- Category: Misc / Experiments
- Primary language: AutoHotkey
- Size/File count: 1635 KB / 9 files
- Features: None
- Focus: Auto Hot key scripts for automation in windows Change_all_Titles.ahk renames all windows to your name of choice Rename_allfiles.ahk renames all files in a entered directory set all windows transparent and on top.ahk s...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 167. `Winclone`

- URL: https://github.com/kai9987kai/Winclone
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 55 KB / 5 files
- Features: webapp, html
- Focus: **Winclone** is a single-file **Windows-style desktop UI clone** built in plain **HTML/CSS/JS** (with jQuery + jQuery UI for drag/drop). It’s a front-end sandbox for experimenting with a “desktop in the browser” exper...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template

### 168. `wordlist`

- URL: https://github.com/kai9987kai/wordlist
- Category: Misc / Experiments
- Primary language: Unknown
- Size/File count: 21511 KB / 9 files
- Features: None
- Focus: Collection of some common wordlists such as RDP password, user name list, ssh password wordlist for brute force. The following is an alphabetical list of IP camera manufacturers and their default usernames and passwor...
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build

### 169. `YouTubeVideoShearchChromeExtension`

- URL: https://github.com/kai9987kai/YouTubeVideoShearchChromeExtension
- Category: Web Apps / Generators
- Primary language: HTML
- Size/File count: 95 KB / 8 files
- Features: webapp, html, javascript
- Focus: A chrome extensions for searching my YouTube channels videos video
- Strengths: README/doc present
- Risks: no obvious tests
- Opportunities: add GitHub Actions for smoke tests/build; deploy preset + environment template; package scripts and lockfile for reproducible frontend builds
