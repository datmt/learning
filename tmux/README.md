# Tmux Ninja Cheat Sheet 🥷

## What is tmux?

**tmux** (terminal multiplexer) is a powerful command-line tool that allows you to create and manage multiple terminal sessions within a single terminal window. Think of it as a "window manager" for your terminal - it lets you split your screen into multiple panes, create multiple windows (like tabs), and most importantly, run terminal sessions that persist even when you disconnect.

## Core Concept: Session Persistence

The game-changing feature of tmux is **session persistence**. When you run commands in a regular terminal and close it, everything stops. With tmux, your sessions continue running in the background - you can disconnect, shut down your computer, reconnect hours later, and pick up exactly where you left off.

## Primary Use Cases

### 1. **Remote Server Management**
```bash
# Connect to server, start tmux session
ssh user@server
tmux new -s work

# Start long-running processes
./deploy-script.sh
python train_model.py

# Detach and disconnect (Ctrl-b d)
# Later: reconnect and reattach
ssh user@server
tmux attach -t work
# Everything is still running!
```

**Why this matters**: Network connections drop, laptops sleep, but your work continues uninterrupted on the server.

### 2. **Development Environment Organization**
```bash
# Create a project workspace
tmux new -s myproject
# Window 1: Code editor
# Window 2: Server running
# Window 3: Tests + logs
# Window 4: Database console
```

Instead of juggling multiple terminal windows/tabs, everything is organized in one persistent session.

### 3. **Multitasking and Workflow Management**
```bash
# Split your terminal into multiple panes:
# - Top: editing code
# - Bottom left: running tests
# - Bottom right: monitoring logs
# All visible simultaneously
```

### 4. **Long-Running Processes**
Perfect for tasks that take hours or days:
- Machine learning model training
- Large database migrations
- Backup operations
- Continuous integration builds
- File transfers

### 5. **Pair Programming / Screen Sharing**
Multiple users can attach to the same tmux session and collaborate in real-time, seeing the same terminal output and being able to type.

## Why Master tmux?

### **Productivity Gains**
- **No context switching**: Everything related to a project stays in one session
- **Instant environment recreation**: Jump back into complex multi-terminal setups instantly
- **Reduced cognitive load**: No need to remember what was running where

### **Reliability & Peace of Mind**
- **Network resilience**: SSH connections can drop without losing work
- **System reliability**: Processes survive terminal crashes, accidental window closures
- **Power management**: Laptop can sleep/hibernate without stopping server work

### **Professional Efficiency**
- **Server administration**: Essential for managing remote systems professionally
- **DevOps workflows**: Managing multiple services, monitoring, deployments simultaneously
- **Data science**: Long-running notebooks, experiments, data processing pipelines

### **Resource Optimization**
- **Single SSH connection**: One connection to server, multiple sessions inside
- **Memory efficient**: Less overhead than multiple terminal emulators
- **Bandwidth saving**: Only one connection for multiple "terminals"

## Real-World Scenarios

### **Scenario 1: Full-Stack Developer**
```bash
Session "webapp":
├── Window 1: Frontend (React dev server)
├── Window 2: Backend (Node.js API)
├── Window 3: Database (MongoDB shell)
└── Window 4: Testing (Jest watch mode)
```

### **Scenario 2: DevOps Engineer**
```bash
Session "monitoring":
├── Pane 1: Server logs (tail -f)
├── Pane 2: Resource monitoring (htop)
├── Pane 3: Network monitoring
└── Pane 4: Command prompt for interventions
```

### **Scenario 3: Data Scientist**
```bash
Session "ml-project":
├── Window 1: Jupyter notebook server
├── Window 2: Training script (running for 12 hours)
├── Window 3: Data preprocessing
└── Window 4: Model evaluation
```

## Learning Investment vs. Return

**Initial time investment**: ~2-4 hours to learn basics
**Mastery time**: ~2-3 weeks of regular use
**Lifetime productivity gain**: Immeasurable

The learning curve is gentle - you can be productive with just 5-6 commands, then gradually add advanced features.

## Who Should Master tmux?

### **Must-have for:**
- Software developers (especially backend/full-stack)
- DevOps engineers and system administrators
- Data scientists and researchers
- Anyone who works with remote servers regularly

### **Very valuable for:**
- Command-line enthusiasts
- People who run long-running processes
- Anyone juggling multiple terminal-based tasks
- Students learning programming/system administration

### **Less critical for:**
- Pure GUI developers (though still useful)
- People who rarely use terminals
- Windows-only users (though WSL makes it relevant)

## The Bottom Line

tmux transforms your terminal from a simple command interface into a powerful, persistent workspace. Once you experience the freedom of detachable sessions and organized multi-pane layouts, working without tmux feels like coding without syntax highlighting - technically possible, but unnecessarily limiting.

The investment in learning tmux pays dividends every single day you use a terminal, making it one of the highest-ROI tools you can master as a developer or system administrator.


## Essential Concepts

**Session**: A collection of windows, can be detached/reattached
**Window**: Like a tab, contains one or more panes
**Pane**: Individual terminal within a window
**Prefix Key**: Default `Ctrl-b` (customizable to `Ctrl-a`)

---

## Session Management

### Basic Session Operations
```bash
# Start new session
tmux
tmux new-session
tmux new -s session-name

# List sessions
tmux ls
tmux list-sessions

# Attach to session
tmux attach
tmux a -t session-name

# Detach from session
Ctrl-b d

# Kill session
tmux kill-session -t session-name
tmux kill-server  # Kill all sessions
```

### Advanced Session Tricks
```bash
# Create named session with specific window
tmux new -s work -n editor

# Attach or create session
tmux new -A -s main

# Session with multiple windows
tmux new -s dev \; new-window -n logs \; new-window -n tests

# Rename session
Ctrl-b $
```

---

## Window Management

### Basic Window Commands
```bash
Ctrl-b c        # Create new window
Ctrl-b n        # Next window
Ctrl-b p        # Previous window
Ctrl-b 0-9      # Switch to window by number
Ctrl-b w        # List and select windows
Ctrl-b ,        # Rename current window
Ctrl-b &        # Kill current window
Ctrl-b f        # Find window by name
```

### Window Ninja Moves
```bash
Ctrl-b .        # Move window (prompt for new number)
Ctrl-b '        # Jump to window by number (prompt)
Ctrl-b l        # Last window (toggle between two)

# Move window to another session
Ctrl-b . then type: session-name:index
```

---

## Pane Management

### Basic Pane Operations
```bash
Ctrl-b %        # Split vertically (left/right)
Ctrl-b "        # Split horizontally (top/bottom)
Ctrl-b arrow    # Navigate between panes
Ctrl-b o        # Cycle through panes
Ctrl-b ;        # Toggle to last active pane
Ctrl-b x        # Kill current pane
Ctrl-b !        # Break pane into new window
```

### Pane Ninja Techniques
```bash
Ctrl-b q        # Show pane numbers
Ctrl-b q 0-9    # Jump to pane by number
Ctrl-b {        # Move pane left
Ctrl-b }        # Move pane right
Ctrl-b Ctrl-o   # Rotate panes
Ctrl-b Alt-o    # Rotate panes other direction

# Resize panes
Ctrl-b Ctrl-arrow   # Resize by 1
Ctrl-b Alt-arrow    # Resize by 5

# Zoom pane (toggle fullscreen)
Ctrl-b z

# Join pane from another window
Ctrl-b : then join-pane -s window.pane
```

### Pane Layouts
```bash
Ctrl-b Space    # Cycle through layouts
Ctrl-b Alt-1    # Even horizontal
Ctrl-b Alt-2    # Even vertical
Ctrl-b Alt-3    # Main horizontal
Ctrl-b Alt-4    # Main vertical
Ctrl-b Alt-5    # Tiled
```

---

## Copy Mode (Vim-style Navigation)

### Enter and Navigate Copy Mode
```bash
Ctrl-b [        # Enter copy mode
q               # Exit copy mode
g               # Go to top
G               # Go to bottom
/               # Search forward
?               # Search backward
n               # Next search result
N               # Previous search result
```

### Selection and Copying
```bash
Space           # Start selection
Enter           # Copy selection
v               # Toggle selection mode
V               # Line selection mode
Ctrl-v          # Block selection mode

# Paste
Ctrl-b ]        # Paste most recent buffer
Ctrl-b =        # Choose buffer to paste
```

### Advanced Copy Mode
```bash
# Navigate like Vim
h j k l         # Move cursor
w b             # Word forward/backward
0 $             # Beginning/end of line
Ctrl-f Ctrl-b   # Page forward/backward
```

---

## Command Mode & Scripting

### Command Mode Basics
```bash
Ctrl-b :        # Enter command mode

# Useful commands
:new-window -n name "command"
:split-window -h "command"
:send-keys "text" Enter
:select-window -t name
:kill-session -t name
```

### Scripting Examples
```bash
# Create development environment
tmux new -s dev -d
tmux send-keys -t dev "cd ~/project" Enter
tmux split-window -t dev -h
tmux send-keys -t dev:0.1 "vim ." Enter
tmux new-window -t dev -n server
tmux send-keys -t dev:server "npm start" Enter
tmux select-window -t dev:0
tmux attach -t dev
```

---

## Configuration Ninja (.tmux.conf)

### Essential Customizations
```bash
# Change prefix key to Ctrl-a
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# Enable mouse support
set -g mouse on

# Start windows and panes at 1
set -g base-index 1
set -g pane-base-index 1

# Reload config
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# Better splitting
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
```

### Advanced Configuration
```bash
# Vim-style pane navigation
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# Vi mode
set-window-option -g mode-keys vi
bind-key -T copy-mode-vi 'v' send -X begin-selection
bind-key -T copy-mode-vi 'y' send -X copy-selection

# Status bar customization
set -g status-position bottom
set -g status-bg colour234
set -g status-fg colour137
set -g status-left '#[fg=colour233,bg=colour241,bold] #S '
set -g status-right '#[fg=colour233,bg=colour241,bold] %d/%m #[fg=colour233,bg=colour245,bold] %H:%M:%S '

# Pane borders
set -g pane-border-style fg=colour238
set -g pane-active-border-style fg=colour208
```

---

## Workflow Ninja Patterns

### Development Environment Setup
```bash
# Quick dev session
alias tdev='tmux new -s dev -d; tmux split-window -t dev -h; tmux send-keys -t dev:0.0 "vim ." Enter; tmux send-keys -t dev:0.1 "git status" Enter; tmux attach -t dev'

# Multi-project workspace
tmux new -s workspace -d
tmux new-window -t workspace -n project1 -c ~/project1
tmux new-window -t workspace -n project2 -c ~/project2
tmux new-window -t workspace -n monitoring
tmux split-window -t workspace:monitoring -h "htop"
```

### Session Templates
```bash
# Create session template script
#!/bin/bash
SESSION="myproject"
tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s $SESSION -x 120 -y 30
  
  # Window 1: Editor
  tmux rename-window -t $SESSION:1 'editor'
  tmux send-keys -t $SESSION:1 'cd ~/project && vim' Enter
  
  # Window 2: Server
  tmux new-window -t $SESSION:2 -n 'server'
  tmux send-keys -t $SESSION:2 'cd ~/project && npm start' Enter
  
  # Window 3: Tests
  tmux new-window -t $SESSION:3 -n 'tests'
  tmux send-keys -t $SESSION:3 'cd ~/project' Enter
  tmux split-window -t $SESSION:3 -h
  tmux send-keys -t $SESSION:3.1 'cd ~/project && npm test -- --watch' Enter
  
  tmux select-window -t $SESSION:1
fi

tmux attach -t $SESSION
```

---

## Power User Commands

### Buffer Management
```bash
Ctrl-b #        # List paste buffers
Ctrl-b =        # Choose buffer to paste
:show-buffer    # Display buffer contents
:save-buffer ~/buffer.txt  # Save buffer to file
:load-buffer ~/file.txt    # Load file into buffer
```

### Advanced Navigation
```bash
:find-window pattern    # Find windows matching pattern
:list-commands         # Show all available commands
:list-keys            # Show all key bindings

# Jump between sessions
Ctrl-b s              # Session tree
Ctrl-b (              # Previous session  
Ctrl-b )              # Next session
```

### Synchronization
```bash
# Synchronize panes (type in all panes simultaneously)
:setw synchronize-panes on
:setw synchronize-panes off

# Send commands to all panes in window
:send-keys -t window-name "command" Enter
```

---

## Troubleshooting & Tips

### Common Issues
```bash
# Fix "sessions should be nested with care" warning
export TMUX=

# 256 color support
export TERM=screen-256color

# Fix clipboard on macOS
set -g default-command "reattach-to-user-namespace -l $SHELL"
```

### Performance Tips
```bash
# Increase history limit
set -g history-limit 10000

# Faster key repetition
set -s escape-time 0

# Aggressive resize (useful for shared sessions)
setw -g aggressive-resize on
```

---

## Quick Reference Card

| Action | Key | Command |
|--------|-----|---------|
| New session | | `tmux new -s name` |
| Attach | | `tmux a -t name` |
| Detach | `Ctrl-b d` | |
| New window | `Ctrl-b c` | |
| Split horizontal | `Ctrl-b "` | |
| Split vertical | `Ctrl-b %` | |
| Navigate panes | `Ctrl-b arrows` | |
| Zoom pane | `Ctrl-b z` | |
| Copy mode | `Ctrl-b [` | |
| Paste | `Ctrl-b ]` | |
| Command mode | `Ctrl-b :` | |
| Kill pane | `Ctrl-b x` | |
| List sessions | | `tmux ls` |

---

## Ninja Aliases

Add these to your shell config for maximum efficiency:

```bash
alias t='tmux'
alias ta='tmux attach'
alias tls='tmux list-sessions'
alias tat='tmux attach -t'
alias tns='tmux new-session -s'
alias tks='tmux kill-session -t'

# Quick attach or create
ta() { tmux attach -t "$1" 2>/dev/null || tmux new -s "$1"; }
```

**Pro Tip**: Start with the basics, then gradually incorporate advanced features. The key to becoming a tmux ninja is consistent daily use and incrementally building your configuration!