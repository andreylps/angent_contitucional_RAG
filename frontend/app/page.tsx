"use client";

import { useState, FormEvent, useRef, useEffect } from "react";

// Define a estrutura de uma mensagem no chat
interface Message {
  text: string;
  isUser: boolean;
}

export default function Home() {
  // Estado para armazenar a lista de mensagens
  const [messages, setMessages] = useState<Message[]>([]);
  // Estado para armazenar o que o usuário está digitando
  const [input, setInput] = useState("");
  // Estado para controlar o status de carregamento (quando a IA está "pensando")
  const [isLoading, setIsLoading] = useState(false);

  // Referência para o final da lista de mensagens, para rolar automaticamente
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Efeito para rolar para a última mensagem sempre que a lista de mensagens mudar
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Função chamada quando o usuário envia o formulário
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault(); // Impede o recarregamento da página
    if (!input.trim() || isLoading) return; // Não faz nada se o input estiver vazio ou se já estiver carregando

    const userMessage: Message = { text: input, isUser: true };
    // Adiciona a mensagem do usuário à lista e limpa o input
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // Faz a requisição POST para a nossa API FastAPI
      const response = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }),
      });

      if (!response.ok) {
        throw new Error("A resposta da rede não foi 'ok'");
      }

      const data = await response.json();

      // Cria a mensagem de resposta da IA
      const aiMessage: Message = { text: data.final_answer, isUser: false };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error("Erro ao buscar resposta da API:", error);
      const errorMessage: Message = {
        text: "Desculpe, ocorreu um erro ao conectar com o assistente. Tente novamente.",
        isUser: false,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      // Independentemente de sucesso ou erro, para de carregar
      setIsLoading(false);
    }
  };

  return (
    <main className="flex flex-col h-screen max-w-4xl mx-auto p-4">
      <h1 className="text-3xl font-bold text-center mb-4">
        RAG Jurídico Assistant
      </h1>

      {/* Área de exibição das mensagens */}
      <div className="flex-1 overflow-y-auto p-4 bg-gray-800 rounded-lg mb-4 space-y-4">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${msg.isUser ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`p-3 rounded-lg max-w-lg ${
                msg.isUser
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 text-gray-200"
              }`}
            >
              {/* Usamos 'whitespace-pre-wrap' para respeitar as quebras de linha da resposta */}
              <p className="whitespace-pre-wrap">{msg.text}</p>
            </div>
          </div>
        ))}
        {/* Exibe um indicador de "pensando..." */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="p-3 rounded-lg bg-gray-700 text-gray-400">
              <p>Pensando...</p>
            </div>
          </div>
        )}
        {/* Elemento invisível para o qual rolamos a tela */}
        <div ref={messagesEndRef} />
      </div>

      {/* Formulário de input */}
      <form onSubmit={handleSubmit} className="flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Digite sua pergunta jurídica..."
          className="flex-1 p-3 bg-gray-700 border border-gray-600 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        />
        <button
          type="submit"
          className="bg-blue-600 text-white p-3 rounded-r-lg hover:bg-blue-700 disabled:bg-blue-400"
          disabled={isLoading}
        >
          Enviar
        </button>
      </form>
    </main>
  );
}
