"""
Analizador Big Five con Gemini
Lee un JSON ya scrapeado y genera el análisis de personalidad.

Uso: python analizar.py <archivo.json>
Ejemplo: python analizar.py auronplay_instagram.json

Requiere: pip install google-genai
"""

import json
import sys
from pathlib import Path
from google import genai
from google.genai import types


# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"


# ──────────────────────────────────────────────────────────────
# CARGAR API KEY DESDE apis.json
# ──────────────────────────────────────────────────────────────
def load_gemini_key(filepath: str = "apis.json") -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {filepath}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key in ("gemini", "gemini_api_key", "GEMINI_API_KEY", "api_key"):
        if key in data and data[key]:
            return data[key]
    raise KeyError(f"No se encontró la key de Gemini en {filepath}")


# ──────────────────────────────────────────────────────────────
# PREPARAR RESUMEN DEL JSON PARA GEMINI
# ──────────────────────────────────────────────────────────────
def preparar_resumen(data: dict) -> str:
    username  = data.get("username", "")
    bio       = data.get("bio", "")
    followers = data.get("followers", "")
    following = data.get("following", "")
    posts     = data.get("posts", [])

    lineas = [
        f"PERFIL DE INSTAGRAM: @{username}",
        f"Bio: {bio}" if bio else "Bio: (sin bio)",
        f"Seguidores: {followers}",
        f"Seguidos: {following}",
        f"Total posts analizados: {len(posts)}",
        "",
        "═══ PUBLICACIONES ═══",
    ]

    for i, post in enumerate(posts, 1):
        fecha    = (post.get("date") or "")[:10]
        location = post.get("location", "")
        caption  = post.get("caption", "")
        comments = post.get("comments", [])

        lineas.append(f"\n[Post {i}] Fecha: {fecha}")
        if location:
            lineas.append(f"  Ubicación: {location}")
        lineas.append(f"  Caption: {caption}" if caption else "  Caption: (sin texto)")

        if comments:
            lineas.append(f"  Comentarios ({len(comments)}):")
            for c in comments[:30]:
                lineas.append(f"    @{c.get('author','?')}: {c.get('text','')}")
        else:
            lineas.append("  Comentarios: (ninguno)")

    return "\n".join(lineas)


# ──────────────────────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────────────────────
def construir_prompt(resumen: str) -> str:
    return f"""Eres un psicólogo experto en análisis de personalidad digital.
Analiza el siguiente perfil de Instagram y genera un análisis de personalidad completo.

{resumen}

═══════════════════════════════════════════════════════════════
INSTRUCCIONES:

1. MODELO BIG FIVE (OCEAN) — puntúa del 1 al 10 y justifica con ejemplos del perfil:
   - O — Apertura a la experiencia
   - C — Responsabilidad
   - E — Extraversión
   - A — Amabilidad
   - N — Neuroticismo

2. PATRONES DE COMPORTAMIENTO DIGITAL:
   - Frecuencia y estilo de publicación
   - Tipo de contenido preferido
   - Relación con su audiencia

3. ESTILO DE COMUNICACIÓN:
   - Tono general (cercano, distante, humorístico, serio)
   - Cómo se expresa en captions y respuestas

4. MOTIVACIONES CENTRALES:
   - Qué busca proyectar
   - Necesidades psicológicas detectadas

5. RESUMEN EJECUTIVO — perfil de personalidad en 3-4 líneas

6. ADVERTENCIA — limitaciones del análisis basado solo en datos públicos

Responde en español. Sé específico y usa ejemplos concretos del perfil.
═══════════════════════════════════════════════════════════════
"""


# ──────────────────────────────────────────────────────────────
# LLAMAR A GEMINI
# ──────────────────────────────────────────────────────────────
def analizar(data: dict) -> str:
    api_key  = load_gemini_key()
    client   = genai.Client(api_key=api_key)
    resumen  = preparar_resumen(data)
    prompt   = construir_prompt(resumen)

    print("🤖 Analizando con Gemini...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4096,
        ),
    )
    return response.text


# ──────────────────────────────────────────────────────────────
# MOSTRAR Y GUARDAR
# ──────────────────────────────────────────────────────────────
def mostrar_y_guardar(username: str, analisis: str):
    sep = "═" * 60
    output = f"""
{sep}
  ANÁLISIS DE PERSONALIDAD BIG FIVE
  Perfil: @{username}
{sep}

{analisis}

{sep}
"""
    print(output)

    out_file = f"{username}_bigfive.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"✅ Análisis guardado en: {out_file}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Uso: python analizar.py <archivo.json>")
        print("Ejemplo: python analizar.py auronplay_instagram.json")
        sys.exit(1)

    json_file = sys.argv[1]

    print(f"📂 Cargando {json_file} ...")
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    username = data.get("username", "perfil")
    print(f"📊 Perfil: @{username}")
    print(f"   Posts: {len(data.get('posts', []))}")
    total_c = sum(len(p.get('comments', [])) for p in data.get('posts', []))
    print(f"   Comentarios totales: {total_c}")

    analisis = analizar(data)
    mostrar_y_guardar(username, analisis)


if __name__ == "__main__":
    main()