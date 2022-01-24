package ifmo.mmmlpmsw.cloud.bot

import org.telegram.telegrambots.bots.TelegramLongPollingBot
import org.telegram.telegrambots.meta.api.methods.AnswerInlineQuery
import org.telegram.telegrambots.meta.api.methods.send.SendMessage
import org.telegram.telegrambots.meta.api.objects.Update
import org.telegram.telegrambots.meta.api.objects.inlinequery.InlineQuery
import org.telegram.telegrambots.meta.api.objects.inlinequery.result.InlineQueryResult
import org.telegram.telegrambots.meta.api.objects.inlinequery.result.InlineQueryResultPhoto
import org.telegram.telegrambots.meta.exceptions.TelegramApiException
import java.net.URLEncoder
import java.util.*


class NeuralCatBot: TelegramLongPollingBot() {

    companion object {

        private const val CACHE_TIME = 0
        private const val URL = "https://app-catbot.azurewebsites.net"
    }

    override fun getBotToken(): String = CatBotProperties.botToken
    override fun getBotUsername(): String = CatBotProperties.botUsername

    override fun onUpdateReceived(update: Update?) {
//        if (update?.message != null && update.message.text != null) {
//            val message: String? = update.message.text
//            sendMsg(update.message.chatId.toString(), message)
//        }

        if (update!!.hasInlineQuery()) {
            processQuery(update.inlineQuery)
        }
    }

    @Synchronized
    fun sendMsg(chatId: String?, s: String?) {
        val message = SendMessage.builder().chatId(chatId!!).text("$s aaaaaaaaaa").build()
        message.enableMarkdown(true)

        try {
            execute(message)
        } catch (e: TelegramApiException) {
            e.printStackTrace()
        }

        // TODO add commands for /emotion and /cat
    }

    @Synchronized
    fun processQuery(inlineQuery: InlineQuery) {
        try {
            if (inlineQuery.query.isNotEmpty()) {
                execute(convertResultsToResponse(inlineQuery))
            }
        } catch (e: TelegramApiException) {
            e.printStackTrace()
        }
    }

    private fun convertResultsToResponse(inlineQuery: InlineQuery): AnswerInlineQuery =
        AnswerInlineQuery().apply {
            inlineQueryId = inlineQuery.id
            results = convertResults(inlineQuery.query)
            cacheTime = CACHE_TIME
        }

    private fun convertResults(text: String): List<InlineQueryResult> {
        val id = UUID.randomUUID().toString()

        val url = "$URL/?text=${URLEncoder.encode(text, "utf-8")}&id=$id"
        val image = InlineQueryResultPhoto(id, url)
        image.thumbUrl = url

        return listOf(image)
    }
}
