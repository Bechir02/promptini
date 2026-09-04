"""Core package — shared logic for Prompt Forge.

Single source of truth for constants, task detection, configuration, and
retrieval fusion. Importing this package must NOT pull in heavy ML deps
(torch / lancedb / sentence-transformers) so the pure logic stays testable.
"""
