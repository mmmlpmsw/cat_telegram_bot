package ifmo.mmmlpmsw.cloud

import org.telegram.telegrambots.api.methods.send.SendMessage
import org.telegram.telegrambots.api.objects.Update
import org.telegram.telegrambots.bots.TelegramLongPollingBot
import org.telegram.telegrambots.exceptions.TelegramApiException


class NeuralCatBot: TelegramLongPollingBot() {

    override fun getBotToken(): String = CatBotProperties.botToken
    override fun getBotUsername(): String = CatBotProperties.botUsername
    override fun onUpdateReceived(update: Update?) {
        val message: String? = update?.message?.text
        sendMsg(update?.message?.chatId.toString(), message)
    }

    @Synchronized
    fun sendMsg(chatId: String?, s: String?) {
        val sendMessage = SendMessage()
        sendMessage.enableMarkdown(true)
        sendMessage.chatId = chatId
        sendMessage.text = "$s aaaaaaaaaa"
        try {
            sendMessage(sendMessage)
        } catch (e: TelegramApiException) {
            e.printStackTrace()
        }
    }

}