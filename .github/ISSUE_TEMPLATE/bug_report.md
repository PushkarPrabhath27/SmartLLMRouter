---
name: Bug Report
about: Create a report to help us improve SmartRoute
title: "[BUG] "
labels: ["bug"]
assignees: ""
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps or minimal runnable Python code snippet to reproduce the behavior:
```python
import asyncio
from smartroute import Router


async def main():
    router = Router()
    ...


asyncio.run(main())
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment & Setup:**
- OS: [e.g. Linux Ubuntu 22.04, macOS Sonoma, Windows 11]
- Python Version: [e.g. 3.10.12, 3.11.8, 3.12.2]
- SmartRoute Version: [e.g. 0.1.0]
- Providers Configured: [e.g. Groq, OpenAI, Anthropic]

**Routing Decisions & Logs**
If applicable, include the `RoutingMeta.reason` or relevant log snippets.
