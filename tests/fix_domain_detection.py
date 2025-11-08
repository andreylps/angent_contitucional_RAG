# scripts/fix_domain_detection.py
#!/usr/bin/env python3  # noqa: EXE005
"""
Corrige urgentemente o domain detector
"""

import os


def fix_domain_detector():
    """Corrige o domain_detector.py"""
    domain_detector_path = "src/utils/domain_detector.py"

    if not os.path.exists(domain_detector_path):  # noqa: PTH110
        print("❌ domain_detector.py não encontrado!")
        return False

    # Lê o arquivo atual
    with open(domain_detector_path, encoding="utf-8") as f:  # noqa: PTH123
        content = f.read()

    # Encontra a seção de NON_LEGAL_KEYWORDS ou adiciona
    if "NON_LEGAL_KEYWORDS" not in content:  # noqa: SIM102
        # Adiciona após LEGAL_KEYWORDS
        if "LEGAL_KEYWORDS" in content:
            # Encontra o fechamento de LEGAL_KEYWORDS e adiciona depois
            lines = content.split("\n")
            new_lines = []
            legal_keywords_found = False

            for line in lines:
                new_lines.append(line)
                if "LEGAL_KEYWORDS" in line and not legal_keywords_found:
                    legal_keywords_found = True
                elif legal_keywords_found and line.strip() == "}":
                    # Adiciona NON_LEGAL_KEYWORDS após fechamento
                    new_lines.append("")
                    new_lines.append("    # 🔴 PALAVRAS-CHAVE DE NÃO-JURÍDICO")
                    new_lines.append("    NON_LEGAL_KEYWORDS = {")
                    new_lines.append(
                        "        'culinaria': ['receita', 'bolo', 'comida', 'culinária', 'cozinhar', 'ingredientes', 'modo de preparo'],"  # noqa: E501
                    )
                    new_lines.append(
                        "        'ciencia': ['física quântica', 'química', 'biologia', 'matemática', 'ciência'],"  # noqa: E501
                    )
                    new_lines.append(
                        "        'culinaria_molecular': ['culinária molecular', 'gastronomia molecular'],"  # noqa: E501
                    )
                    new_lines.append(
                        "        'culinaria_geral': ['cozinha', 'assar', 'forno', 'temperatura culinária', 'culinária']"  # noqa: E501
                    )
                    new_lines.append("    }")
                    new_lines.append("")
                    legal_keywords_found = False

            content = "\n".join(new_lines)

    # Escreve o arquivo corrigido
    with open(domain_detector_path, "w", encoding="utf-8") as f:  # noqa: PTH123
        f.write(content)

    print("✅ domain_detector.py corrigido!")
    return True


if __name__ == "__main__":
    fix_domain_detector()
