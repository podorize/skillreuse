# 6. Conclusão e trabalhos futuros

Este trabalho investigou se características observáveis de uma skill e de seu repositório se associam ao reúso entre donos no GitSkills e quanto dessa previsibilidade é artefato da construção do dataset. Três conclusões se sustentam.

**Primeiro**, o GitSkills tem vieses de construção que afetam estudos preditivos. A regra que escolhe o representante faz a probabilidade de ele estar em `.claude/skills/` crescer de 14% para 92% com o número de cópias, enquanto a fração real de cópias nessa pasta fica em 5–13%. Isso torna a localização e o histórico de commits preditores espúrios de reúso. Além disso, 28% das skills reutilizadas existem em um único dono, e poucas contas concentram a amostra, de modo que contar repositórios superestima a difusão e o desempenho depende de poucos donos (AUC de 0,615 a 0,747 conforme o tratamento).

**Segundo**, livres desses artefatos, os atributos preveem o reúso entre donos com um sinal modesto, confirmado em dados novos: no holdout disjunto por dono, o Random Forest atinge AUC 0,731 (IC95% 0,716–0,745) e o HistGradientBoosting reduz o RMSLE de `owner_count` em 6% frente à média (0,625 contra 0,664). Skill e repositório contribuem de forma complementar, e o sinal se associa sobretudo à visibilidade do repositório.

**Terceiro**, o limite parece estar nos dados e não nos modelos: seis famílias empatam e o tuning não ajuda. As afirmações são associativas e não identificam o que torna uma skill mais ou menos copiada, nem sua qualidade.

**Trabalhos futuros.** (i) Modelar no nível da cópia, com informação anterior ao reúso, para usar atributos de todas as cópias sem vazamento. (ii) Agrupar cópias quase idênticas para mitigar a fragmentação do alvo por hash exato. (iii) Replicar com instantâneos posteriores do GitSkills e estudar a dinâmica temporal da difusão. (iv) Investigar a licença e o tamanho do catálogo como indicadores de clones de catálogos. (v) Avaliar representações de texto (por exemplo, embeddings) dentro de classes de localização fixas. (vi) Propor aos mantenedores do GitSkills indicadores que separem a escolha do representante dos atributos de repositório.
