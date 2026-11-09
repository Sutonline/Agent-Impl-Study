---
name: code-reviewer
description: Structured code review checklist for auditing Python files for bugs, security risks, and style.
---

# Code Reviewer Skill

When asked to review code, follow these exact steps:
1. Read the target file using `read_file`.
2. Evaluate the code against three dimensions:
   - **[BUGS]**: Edge cases, division by zero, off-by-one errors, or type mismatches.
   - **[SECURITY]**: Hardcoded secrets, command injection, or unsafe `eval`.
   - **[STYLE]**: Compliance with workspace `AGENTS.md` naming conventions.
3. Output the review strictly with the header `## [SKILL:CODE-REVIEW-REPORT]` followed by numbered findings.
