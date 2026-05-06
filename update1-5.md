# RAG-Lab Update 1.5

## Descripción
Nueva fase de desarrollo para el sistema RAG-Lab. Este documento registra las mejoras, correcciones y nuevas funcionalidades que se implementan en esta versión.

## Contexto
- Fecha de inicio: 2026-05-06
- Versión objetivo: v1.5
- Dependencias: Ninguna nueva requerida

## Tareas Pendientes de Updates Anteriores
- [ ] Re-ingesta completa (bloqueada por falta de VRAM — SGLang ocupando GPU)
- [ ] Verificar que los fixes de Update 1.4 estén aplicados en la base de datos

## Tareas Planificadas
- [ ] Completar re-ingesta del documento principal
- [ ] Verificar recuperación de la sección "Maintenance Agencies"
- [ ] Añadir nuevos features o mejoras (definir con el usuario)

## Tareas Completadas
- [x] Crear gestor de documentos con CLI independiente
- [x] Implementar almacenamiento SQLite con metadatos y etiquetas
- [x] Implementar detección de duplicados por hash
- [x] Implementar borrado de chunks de ChromaDB al eliminar documento
- [x] Escribir tests para todas las funcionalidades (10 tests)

## Archivos Creados
- `rag_lab/doc_manager/__init__.py` — Paquete del gestor
- `rag_lab/doc_manager/doc_store.py` — Almacenamiento SQLite + lógica de etiquetas
- `rag_lab/doc_manager/cli.py` — CLI con Typer + Rich
- `tests/test_doc_manager/test_doc_manager.py` — 10 tests pasando

## CLI del Gestor
```bash
# Iniciar el gestor
python -m rag_lab.doc_manager.cli

# Comandos disponibles:
docs list              # Listar documentos
docs add <archivo>     # Añadir documento (con detección de duplicados)
docs delete <doc_id>   # Eliminar documento y sus chunks de ChromaDB
docs tag <doc_id> <tag>    # Asignar etiqueta
docs untag <doc_id> <tag>  # Quitar etiqueta
docs search <query>    # Buscar por nombre, ruta o etiquetas
docs collections       # Listar todas las etiquetas
docs info <doc_id>     # Información detallada
```

## Notas
- El gestor es independiente del pipeline RAG principal
- Usa SQLite para metadatos y ChromaDB para chunks
- Las etiquetas permiten organizar y filtrar documentos
- La detección de duplicados por hash evita ingestas duplicadas
